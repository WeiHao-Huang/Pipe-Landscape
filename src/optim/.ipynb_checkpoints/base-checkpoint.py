from contextlib import nullcontext
import copy
from pathlib import Path
import time
import yaml

import torch
import wandb

from logger.logger import DynamicsLogger
from optim.weight_averaging import (
    WeightAverager,
    eval_ema,
    eval_wa,
    ExponentialWeightAverager,
)
from .utils import (
    eval,
    get_batch,
    load_checkpoint,
    load_worker_state,
    save_checkpoint,
    save_worker_state,
)
import copy

### 判断在该ckpt是否要存储
# def should_save_checkpoint(curr_iter, phase1_steps, phase2_steps, phase1_interval, phase2_interval, phase3_interval):
#     if curr_iter < phase1_steps:
#         return curr_iter % phase1_interval == 0
#     elif curr_iter == phase1_steps:
#         return True
#     elif phase1_steps < curr_iter < phase2_steps:
#         return (curr_iter - phase1_steps) % phase2_interval == 0
#     elif curr_iter == phase2_steps:
#         return True  # 强制在第 phase2_steps 步保存
#     else:
#         return (curr_iter - phase2_steps) % phase3_interval == 0


def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
):
    not_compiled_model = model
    if cfg.compile:
        print(f"Compiling model ...")
        model = torch.compile(model)

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()

    if cfg.resume_from and (cfg.resume_from2==None or cfg.resume_from3==None):
        # This is a full resume including the model weights, optimizer, state
        # dataloader state, random seed, etc. Not indended for fine tuning or
        # other scenarios where some of these should change.
        print(f"\nResuming Training From {cfg.resume_from}")
        ckpt_dir = Path(cfg.resume_from)
        curr_iter = load_checkpoint(
            model,
            opt,
            scheduler,
            ckpt_dir / "main.pt",
            cfg.device,
        )
        load_worker_state(ckpt_dir)

        """
        Yupeng, 2025.5.19
        加入一个从2D landscape 中选择一个点resume 的方法
        """
    elif cfg.resume_from and cfg.resume_from2 and cfg.resume_from3:
        print(f"\nResuming Training From {cfg.resume_from}, {cfg.resume_from2} ,{cfg.resume_from3}, the alpha, beta is: {cfg.alpha}, {cfg.beta}")
        current_ckpt1 = torch.load(f"{cfg.resume_from}/main.pt",map_location=torch.device('cpu'))
        current_ckpt2 = torch.load(f"{cfg.resume_from2}/main.pt",map_location=torch.device('cpu'))
        current_ckpt3 = torch.load(f"{cfg.resume_from3}/main.pt",map_location=torch.device('cpu'))
        current_ckpt4 = copy.deepcopy(current_ckpt3)
        for key, value in current_ckpt4["model"].items():
            current_ckpt4["model"][key].data.copy_(current_ckpt3["model"][key] + 
                    cfg.alpha*(current_ckpt1["model"][key] - current_ckpt3["model"][key]) + 
                    cfg.beta*(current_ckpt2["model"][key] - current_ckpt3["model"][key]))

        for i in range(len(current_ckpt4['optimizer']['state'])):
            for key,value in current_ckpt4['optimizer']['state'][i].items():
                if key != "step":
                    current_ckpt4['optimizer']['state'][i][key].data.copy_(current_ckpt3['optimizer']['state'][i][key] + 
                    cfg.alpha*(current_ckpt1['optimizer']['state'][i][key] - current_ckpt3['optimizer']['state'][i][key]) + 
                    cfg.beta*(current_ckpt2['optimizer']['state'][i][key] - current_ckpt3['optimizer']['state'][i][key]))

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        model.load_state_dict(current_ckpt4["model"])
        opt.load_state_dict(current_ckpt4["optimizer"])
        scheduler.load_state_dict(current_ckpt4["scheduler"])
        itr = current_ckpt4["itr"]
        curr_iter = itr
        ckpt_dir = Path(cfg.resume_from3)
        load_worker_state(ckpt_dir)
                
    else:
        curr_iter = 0
        # 保存初始模型
        # ckpt_dir = exp_dir / "ckpts" / "initial"
        # if distributed_backend.is_master_process():
        #     save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
        # save_worker_state(ckpt_dir)


    if cfg.weight_average:
        # This does generally not support resuming training, but will work if
        # cfg.wa_interval perfectly divides the iteration number of the chkpt.
        # Otherwise, the first avg will not be correctly computed, with a bias
        # towards the first sample and missing values for earlier iterations.
        weight_averager = WeightAverager(
            not_compiled_model,
            horizon=cfg.wa_horizon,
            interval=cfg.wa_interval,
            save_dir=None if cfg.wa_use_temp_dir else exp_dir / "avgs",
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
            count=curr_iter,
        )

    if cfg.exponential_moving_average:
        ema = ExponentialWeightAverager(
            not_compiled_model,
            interval=cfg.ema_interval,
            decay=cfg.ema_decay,
            warmup=cfg.warmup_steps if cfg.ema_after_warmup else 0,
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
        )

    if distributed_backend.is_master_process() and cfg.log_dynamics:
        with open(cfg.dynamics_logger_cfg, "r") as f:
            dlcfg = yaml.safe_load(f)

        # Hooks into optimizer
        dlogger = DynamicsLogger(
            model, opt, dlcfg, cfg.results_base_folder, wandb=cfg.wandb
        )
        dlogger.iteration = curr_iter


    


    substep = curr_iter * cfg.acc_steps
    train_reader, val_reader = datareaders["train"], datareaders["val"]
    train_reader.set_step(substep)
    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}
    model.train()

    training_start_time = time.time()
    
    while curr_iter <= cfg.iterations:
        
        checkpoint_steps = set(cfg.checkpoint_steps)

        if curr_iter in checkpoint_steps:
            ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
            if distributed_backend.is_master_process():
                save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
            save_worker_state(ckpt_dir)

        # 在我现在的代码中，一般不使用这块
        # Save permanent checkpoint
        if cfg.permanent_ckpt_interval > 0:
            if curr_iter % cfg.permanent_ckpt_interval == 0:
                ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir)

        # 在我现在的代码中，一般不使用这块
        # Save temporary checkpoint for resuming training
        if cfg.latest_ckpt_interval > 0:
            if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == cfg.iterations:
                ckpt_dir = exp_dir / "ckpts" / "latest"
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir)

        ws = distributed_backend.get_world_size()
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size
        epoch = tokens / train_reader.num_tokens
        if (
            curr_iter % cfg.eval_interval == 0
            or curr_iter == cfg.iterations
            or (curr_iter in cfg.full_eval_at)
        ):
            eval_and_log(
                curr_iter,
                epoch,
                model,
                val_reader,
                type_ctx,
                distributed_backend,
                cfg,
                opt,
                full_eval=(curr_iter in cfg.full_eval_at),
            )

            if curr_iter > cfg.wa_interval and cfg.weight_average:
                eval_wa(
                    curr_iter,
                    not_compiled_model,
                    weight_averager,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )
            if cfg.exponential_moving_average:
                eval_ema(
                    curr_iter,
                    not_compiled_model,
                    ema,
                    val_reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

        if curr_iter == cfg.iterations:
            # Save checkpoints and evaluate at final iteration, but no need to train further    
            # 保存最终模型
            final_ckpt_dir = exp_dir / "ckpts" / "final"
            if distributed_backend.is_master_process():
                save_checkpoint(model, opt, scheduler, curr_iter, final_ckpt_dir)
            save_worker_state(final_ckpt_dir)
            break


        # Train model
        t_start = time.perf_counter_ns()
        for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
            x, y = get_batch(train_reader, device=cfg.device)
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_idx,
                    gradient_accumulation_steps=cfg.acc_steps,
                ):
                    outputs = model(x, targets=y)

            loss = outputs["loss"] / cfg.acc_steps
            loss.backward()
            substep += 1

        if cfg.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if cfg.opt == "SFAdamW":
            opt.train()
        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        if cfg.weight_average:
            weight_averager.step(not_compiled_model, distributed_backend.is_master_process())
        if cfg.exponential_moving_average:
            ema.step(not_compiled_model, distributed_backend.is_master_process())
        dt = (time.perf_counter_ns() - t_start) / 1e9

        curr_iter += 1

        if (
            cfg.log_interval
            and curr_iter % cfg.log_interval == 0
            and distributed_backend.is_master_process()  # Only log on master rank
        ):
            #print("check")
            train_loss = loss.detach().cpu().item() * cfg.acc_steps

            current_lrs = [param_group["lr"] for param_group in opt.param_groups]

            elapsed_time = time.time() - training_start_time
            time_per_iter = elapsed_time / curr_iter if curr_iter > 0 else 0
            remaining_iters = cfg.iterations - curr_iter
            estimated_remaining_time = time_per_iter * remaining_iters

            elapsed_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
            remaining_str = time.strftime('%H:%M:%S', time.gmtime(estimated_remaining_time))

            print(
                f"Train: Iter={curr_iter} ({epoch:0.3f} epochs) "
                f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
                f"lr={current_lrs[0]:.2e} "
                f"[{elapsed_str}<{remaining_str}]"
            )

            if cfg.wandb:
                wandb.log(
                    {
                        "iter": curr_iter,
                        "train/loss": train_loss,
                        "train/perplexity": 2.71828**train_loss,
                        "lr": current_lrs[0],
                        "iter_dt": dt
                    }
                )

    return stats
    

def eval_and_log(
    curr_iter,
    epoch,
    model,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    opt,
    full_eval=False,
):

    model.eval()
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return
    if cfg.opt == "SFAdamW":
        opt.eval()

    if curr_iter == cfg.iterations or full_eval:
        max_num_batches = val_reader.num_batches()
    else:
        max_num_batches = cfg.eval_batches

    # to make sure we start from the beginning of the validation set,
    # i.e. repeat the same batches
    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity = eval(
        model,
        val_reader,
        cfg.device,
        max_num_batches=max_num_batches,
        ctx=type_ctx,
        cfg=cfg,
    )

    print(
        f">Eval: Iter={curr_iter} ({epoch:0.3f} epochs) "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )

    if cfg.wandb:
        if curr_iter == cfg.iterations or full_eval:
            logs = {
                "iter": curr_iter,
                "final-val/loss": val_loss,
                "final-val/perplexity": val_perplexity,
                "final-val/acc": val_acc,
            }
        else:
            logs = {
                "iter": curr_iter,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "val/acc": val_acc,
            }

        wandb.log(logs)
        if cfg.eval_seq_prefix != "none" and (
            curr_iter % (cfg.eval_interval * 5) == 0 or curr_iter == cfg.iterations
        ):
            text_table = wandb.Table(columns=["itr", "val-pp", "text"])

            out_str = distributed_backend.get_raw_model(model).generate_from_string(
                cfg.eval_seq_prefix,
                max_new_tokens=40,
                temperature=0.9,
                top_k=None,
            )
            text_table.add_data(curr_iter, val_perplexity, out_str)
            # why a copy? see github.com/wandb/wandb/issues/2981
            wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})
    model.train()
