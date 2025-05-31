# ============================================================================
# 📝 .PHONY 声明（强制目标始终执行，无论文件是否存在）
#
# - init       : 顶层入口任务，可能和某些 init 脚本重名
# - link-all   : 逻辑组合目标，不会生成文件，但可能被误认
# - $(DOTFILES): 每个 dotfile 名字本身就是实际文件（如 ~/.bashrc），
#                若不标为 .PHONY，make 会以为这些文件已存在，从而跳过命令执行
# ============================================================================
.PHONY: init link-all $(DOTFILES)

USER ?= $(shell whoami)
DOTFILES_DIR = /$(USER)/dotfiles
# DOTFILES = .bashrc .ssh/config
DOTFILES := $(shell find $(DOTFILES_DIR) -type f -printf "%P\n")

init: link-all

link-all: $(DOTFILES)

$(DOTFILES):
	@bash -c '\
	src="$(DOTFILES_DIR)/$@"; \
	dest="$$HOME/$@"; \
	if [ -L "$$dest" ]; then \
		ln -sf "$$src" "$$dest"; \
		echo "Replaced existing symlink: $$dest -> $$src"; \
	elif [ -f "$$dest" ]; then \
		mv "$$dest" "$$dest.backup.$$(date +%Y%m%d%H%M%S)"; \
		ln -s "$$src" "$$dest"; \
		echo "Backed up original and linked: $$dest → $$src"; \
	elif [ ! -e "$$dest" ]; then \
		ln -s "$$src" "$$dest"; \
		echo "Linked: $$dest -> $$src"; \
	else \
		echo "Cannot link: $$dest exists but is not a file/symlink)."; \
	fi'
	

# TODO: 添加 unlink 和 status 目标





# ----------------------- To Be Deleted --------------------------------------

link-ssh-config:
	@target="/$(USER)/config/.ssh/config"; \
	link="$$HOME/.ssh/config"; \
	if [ ! -L "$$link" ]; then \
		ln -s "$$target" "$$link"; \
		echo "Linked SSH config: $$link -> $$target"; \
	else \
		ln -sf "$$target" "$$link"; \
		echo "Replacing existing SSH config: $$link -> $$target"; \
	fi

link-bashrc:
	@bash -c '\
	src="/$(USER)/config/.bashrc"; \
	dest="$$HOME/.bashrc"; \
	if [ -L "$$link" ]; then \
		ln -sf "$$target" "$$link"; \
		echo "Replaced existing symlink: $$link -> $$target"; \
	elif [ -f "$$link" ]; then \
		mv "$$link" "$$link.backup"; \
		ln -s "$$target" "$$link"; \
		echo "Backed up original and linked: $$link → $$target"; \
	elif [ ! -e "$$link" ]; then \
		ln -s "$$target" "$$link"; \
		echo "Linked: $$link -> $$target"; \
	else \
		echo "Cannot link: $$link exists (not a file/symlink)."; \
	fi'
	
stage:
	@if [ -z "$(m)" ]; then echo "❌ Missing commit message: pass m=\"your message\""; exit 1; fi
	@git add .
	@git commit -m "$(m)"

push:
	@if [ -z "$(b)" ]; then echo "❌ Missing branch name: pass b=branch-name"; exit 1; fi
	@git push origin "$(b)"
