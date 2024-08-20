 for config_file in evaluation_configs/*
	echo $config_file
	set -x CHRONOS_EVAL_CONFIG (greadlink -f $config_file )
	echo $CHRONOS_EVAL_CONFIG
	python viz_local_models.py
end
