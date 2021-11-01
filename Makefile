format:
	isort main.py gym_gridverse_baselines
	black main.py gym_gridverse_baselines

lint: ## check style with flake8
	flake8 main.py gym_gridverse_baselines
	pylint main.py gym_gridverse_baselines
