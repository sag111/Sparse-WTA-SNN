set terminal cairolatex standalone size 15cm, 15cm
set output 'delta_G_from_nest_and_original.tex'

G_initial_values_in_original_data = '0.0001 0.0006666666666666666 0.0002 0.00125'  # [S]
G_initial_values_in_nest = '0.001 0.01 0.06666666666666667 0.02 0.125 1.0'
original_data_path = '../../../Memristor_STDP/fit/data'
original_data_filename = original_data_path.'/PPX-5_devices.csv'
nest_data_path = '.'
nest_data_filename = nest_data_path.'/delta_G_vs_delta_t_and_G_initial.csv'

# Set input format to csv.
set datafile separator comma
# Do not draw gaps in place of non-plotted points
# (which are skipped because belong to another G_initial).
set datafile missing NaN

set xlabel '$\Delta t$, ms'
set ylabel '$\Delta G$'

Gmax = 1e-2
plot \
	for [i = 1 : words(G_initial_values_in_original_data)] \
		original_data_filename \
		using \
			(column('G_initial') == word(G_initial_values_in_original_data, i) ? column('delta_t') : NaN) \
			:(column('delta_G_avg')/Gmax):(column('delta_G_err')/Gmax) \
		with yerrorbars lt i \
		title '$\Delta G_\mathrm{experimental}$, $G_\mathrm{initial} = '.word(G_initial_values_in_original_data, i).'$', \
	for [i = 1 : words(G_initial_values_in_nest)] \
		nest_data_filename \
		using \
			(column('G_initial') == word(G_initial_values_in_nest, i) ? column('delta_t') : NaN) \
			:(column('delta_G_from_nest')) \
		with lines lt i dt 1 \
		title '$\Delta G_\mathrm{NEST}$, $G_\mathrm{initial} = '.word(G_initial_values_in_nest, i).'$', \
	for [i = 1 : words(G_initial_values_in_nest)] \
		nest_data_filename \
		using \
			(column('G_initial') == word(G_initial_values_in_nest, i) ? column('delta_t') : NaN) \
			:(column('delta_G_from_fit')) \
		with lines lt i lw 5 \
		title '$\Delta G_\mathrm{fitted}$, $G_\mathrm{initial} = '.word(G_initial_values_in_nest, i).'$'
