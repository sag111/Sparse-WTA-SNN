#!/bin/bash

MODULE_NAME=diehl_neuron_module

nestml --module_name $MODULE_NAME --input_path `pwd`

nest_environment_init_script="`nest-config --exec-prefix`/bin/nest_vars.sh"
if ! test -n "`grep $MODULE_NAME ~/.profile`"
then
	cat >> $nest_environment_init_script <<-END_OF_INPUT_TO_CAT

		# Added by $0:
		# Tell NEST to load its additional module
		export NEST_MODULES=$MODULE_NAME${NEST_MODULES:+:$NEST_MODULES}
	END_OF_INPUT_TO_CAT
	cat <<-END_OF_INPUT_TO_CAT
		"NEST_MODULES=$MODULE_NAME${NEST_MODULES:+:$NEST_MODULES}" added to $nest_environment_init_script
		Re-run
		source $nest_environment_init_script for it to take effect.
	END_OF_INPUT_TO_CAT
fi