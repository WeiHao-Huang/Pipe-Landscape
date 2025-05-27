init: link-ssh-config

link-ssh-config:
	@if [ ! -L ~/.ssh/config ]; then \
		ln -s /$(USER)/config/.ssh/config ~/.ssh/config; \
	fi
