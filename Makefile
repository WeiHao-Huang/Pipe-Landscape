init: link-ssh-config

link-ssh-config:
	@if [ ! -L ~/.ssh/config ]; then \
		ln -s /$(USER)/config/.ssh/config ~/.ssh/config; \
		echo "Linked SSH config."; \
	else \
		echo "SSH config already linked."; \
	fi

stage:
	@if [ -z "$(m)" ]; then echo "❌ Missing commit message: pass m=\"your message\""; exit 1; fi
	@git add .
	@git commit -m "$(m)"

push:
	@if [ -z "$(b)" ]; then echo "❌ Missing branch name: pass b=branch-name"; exit 1; fi
	@git push origin "$(b)"
