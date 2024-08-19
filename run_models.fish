 for config_file in evaluation_configs/*
	echo $config_file
	set -x CHRONOS_EVAL_CONFIG (greadlink -f $config_file )
	echo $CHRONOS_EVAL_CONFIG
	python viz_seasonal_naive_eval.py
	python viz_prophet_eval.py
	python viz_arima_eval.py
end
