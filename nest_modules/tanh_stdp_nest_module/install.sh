#!/bin/bash

# The make install step requires this directory,
# even though I am not going to place anything there.

MODULE_NAME=stdptanhmodule

mkdir -p stdp_tanh_module/sli

mkdir -p stdp_tanh_module_build
cd stdp_tanh_module_build

nest_environment_init_script="`nest-config --exec-prefix`/bin/nest_vars.sh"

cmake -Dwith-nest=`which nest-config` ../stdp_tanh_module &&
make &&
mkdir -p ../stdp_tanh_module/sli && # installer requires this dir even if empty
make install &&
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