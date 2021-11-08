format:
	isort render.py plan.py gym_gridverse_baselines
	black render.py plan.py gym_gridverse_baselines

lint: ## check style with flake8
	flake8 render.py plan.py gym_gridverse_baselines
	pylint render.py plan.py gym_gridverse_baselines
