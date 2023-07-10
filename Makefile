.PHONY: push

push:
	git add .
	git commit -m "$(commit)"
	git push

train:
	python train.py --config-path=exps --config-name=$(exp)