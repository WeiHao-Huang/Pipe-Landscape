USER ?= $(shell whoami)

init: link-ssh-config

link-ssh-config:
	@target="/$(USER)/config/.ssh/config"; \
	link="$$HOME/.ssh/config"; \
	if [ ! -L ~/.ssh/config ]; then \
		ln -s "$$target" "$$link"; \
		echo "Linked SSH config: $$link -> $$target"; \
	else \
		ln -sf "$$target" "$$link"; \
		echo "Replacing existing config: $$link -> $$target"; \
	fi

stage:
	@if [ -z "$(m)" ]; then echo "❌ Missing commit message: pass m=\"your message\""; exit 1; fi
	@git add .
	@git commit -m "$(m)"

push:
	@if [ -z "$(b)" ]; then echo "❌ Missing branch name: pass b=branch-name"; exit 1; fi
	@git push origin "$(b)"
