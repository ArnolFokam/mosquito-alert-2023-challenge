.PHONY: push

push:
	git add .
	git commit -m "$(commit)"
	git push

train:
	ulimit -n 2048 && python train.py --config-path=exps --config-name=$(exp)