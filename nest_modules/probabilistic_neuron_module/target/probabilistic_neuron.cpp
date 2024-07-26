// #define DEBUG 1
/*
 *  probabilistic_neuron.cpp
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Generated from NESTML at time: 2024-07-09 06:06:00.027269
**/

// C++ includes:
#include <limits>

// Includes from libnestutil:
#include "numerics.h"

// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "nest_impl.h"
#include "universal_data_logger_impl.h"

// Includes from sli:
#include "dict.h"
#include "dictutils.h"
#include "doubledatum.h"
#include "integerdatum.h"
#include "lockptrdatum.h"

#include "probabilistic_neuron.h"
void
register_probabilistic_neuron( const std::string& name )
{
  nest::register_node_model< probabilistic_neuron >( name );
}

// ---------------------------------------------------------------------------
//   Recordables map
// ---------------------------------------------------------------------------
nest::RecordablesMap<probabilistic_neuron> probabilistic_neuron::recordablesMap_;
namespace nest
{

  // Override the create() method with one call to RecordablesMap::insert_()
  // for each quantity to be recorded.
template <> void RecordablesMap<probabilistic_neuron>::create()
  {
    // add state variables to recordables map
   insert_(probabilistic_neuron_names::_I_syn, &probabilistic_neuron::get_I_syn);

    // Add vector variables  
  }
}
std::vector< std::tuple< int, int > > probabilistic_neuron::rport_to_nestml_buffer_idx =
{
  
  { probabilistic_neuron::EXC_SPIKES, probabilistic_neuron::INH_SPIKES },
};

// ---------------------------------------------------------------------------
//   Default constructors defining default parameters and state
//   Note: the implementation is empty. The initialization is of variables
//   is a part of probabilistic_neuron's constructor.
// ---------------------------------------------------------------------------

probabilistic_neuron::Parameters_::Parameters_()
{
}

probabilistic_neuron::State_::State_()
{
}

// ---------------------------------------------------------------------------
//   Parameter and state extractions and manipulation functions
// ---------------------------------------------------------------------------

probabilistic_neuron::Buffers_::Buffers_(probabilistic_neuron &n):
  logger_(n)
  , spike_inputs_( std::vector< nest::RingBuffer >( NUM_SPIKE_RECEPTORS ) )
  , spike_inputs_grid_sum_( std::vector< double >( NUM_SPIKE_RECEPTORS ) )
{
  // Initialization of the remaining members is deferred to init_buffers_().
}

probabilistic_neuron::Buffers_::Buffers_(const Buffers_ &, probabilistic_neuron &n):
  logger_(n)
  , spike_inputs_( std::vector< nest::RingBuffer >( NUM_SPIKE_RECEPTORS ) )
  , spike_inputs_grid_sum_( std::vector< double >( NUM_SPIKE_RECEPTORS ) )
{
  // Initialization of the remaining members is deferred to init_buffers_().
}

// ---------------------------------------------------------------------------
//   Default constructor for node
// ---------------------------------------------------------------------------

probabilistic_neuron::probabilistic_neuron():ArchivingNode(), P_(), S_(), B_(*this)
{
  init_state_internal_();
  recordablesMap_.create();
  pre_run_hook();
}

// ---------------------------------------------------------------------------
//   Copy constructor for node
// ---------------------------------------------------------------------------

probabilistic_neuron::probabilistic_neuron(const probabilistic_neuron& __n):
  ArchivingNode(), P_(__n.P_), S_(__n.S_), B_(__n.B_, *this) {

  // copy parameter struct P_

  // copy state struct S_
  S_.I_syn = __n.S_.I_syn;

  // copy internals V_
}

// ---------------------------------------------------------------------------
//   Destructor for node
// ---------------------------------------------------------------------------

probabilistic_neuron::~probabilistic_neuron()
{
}

// ---------------------------------------------------------------------------
//   Node initialization functions
// ---------------------------------------------------------------------------
void probabilistic_neuron::calibrate_time( const nest::TimeConverter& tc )
{
  LOG( nest::M_WARNING,
    "probabilistic_neuron",
    "Simulation resolution has changed. Internal state and parameters of the model have been reset!" );

  init_state_internal_();
}
void probabilistic_neuron::init_state_internal_()
{
#ifdef DEBUG
  std::cout << "probabilistic_neuron::init_state_internal_()" << std::endl;
#endif

  const double __resolution = nest::Time::get_resolution().get_ms();  // do not remove, this is necessary for the resolution() function

  recompute_internal_variables();
  // initial values for state variables
    

    S_.I_syn = 0; // as pA
}

void probabilistic_neuron::init_buffers_()
{
#ifdef DEBUG
  std::cout << "probabilistic_neuron::init_buffers_()" << std::endl;
#endif
  // spike input buffers
  get_spike_inputs_().clear();
  get_spike_inputs_grid_sum_().clear();

  B_.logger_.reset();


}

void probabilistic_neuron::recompute_internal_variables(bool exclude_timestep) {
  const double __resolution = nest::Time::get_resolution().get_ms();  // do not remove, this is necessary for the resolution() function

  if (exclude_timestep) {    
  }
  else {    
  }
}
void probabilistic_neuron::pre_run_hook() {
  B_.logger_.init();

  // parameters might have changed -- recompute internals
  recompute_internal_variables();

  // buffers B_
  B_.spike_inputs_.resize(NUM_SPIKE_RECEPTORS);
  B_.spike_inputs_grid_sum_.resize(NUM_SPIKE_RECEPTORS);
}

// ---------------------------------------------------------------------------
//   Update and spike handling functions
// ---------------------------------------------------------------------------


void probabilistic_neuron::update(nest::Time const & origin,const long from, const long to)
{
  const double __resolution = nest::Time::get_resolution().get_ms();  // do not remove, this is necessary for the resolution() function



  for ( long lag = from ; lag < to ; ++lag )
  {
    auto get_t = [origin, lag](){ return nest::Time( nest::Time::step( origin.get_steps() + lag + 1) ).get_ms(); };

    for (long i = 0; i < NUM_SPIKE_RECEPTORS; ++i)
    {
        get_spike_inputs_grid_sum_()[i] = get_spike_inputs_()[i].get_value(lag);
    }

    // NESTML generated code for the update block
  double I_syn_orig = S_.I_syn;
  S_.I_syn += (0.001 * B_.spike_inputs_grid_sum_[EXC_SPIKES - MIN_SPIKE_RECEPTOR]) * 1.0 * 1000.0;
  S_.I_syn -= (0.001 * B_.spike_inputs_grid_sum_[INH_SPIKES - MIN_SPIKE_RECEPTOR]) * 1.0 * 1000.0;
  if (((0) + (1) * nest::get_vp_specific_rng( get_thread() )->drand()) < S_.I_syn)
  {  
    set_spiketime(nest::Time::step(origin.get_steps()+lag+1));
    nest::SpikeEvent se;
    nest::kernel().event_delivery_manager.send(*this, se, lag);
  }
  S_.I_syn = I_syn_orig;
    // voltage logging
    B_.logger_.record_data(origin.get_steps() + lag);
  }
}

// Do not move this function as inline to h-file. It depends on
// universal_data_logger_impl.h being included here.
void probabilistic_neuron::handle(nest::DataLoggingRequest& e)
{
  B_.logger_.handle(e);
}


void probabilistic_neuron::handle(nest::SpikeEvent &e)
{
  assert(e.get_delay_steps() > 0);
  assert( e.get_rport() < B_.spike_inputs_.size() );

  double weight = e.get_weight();
  size_t nestml_buffer_idx = 0;
  if ( weight >= 0.0 )
  {
    nestml_buffer_idx = std::get<0>(rport_to_nestml_buffer_idx[e.get_rport()]);
  }
  else
  {
    nestml_buffer_idx = std::get<1>(rport_to_nestml_buffer_idx[e.get_rport()]);
    if ( nestml_buffer_idx == probabilistic_neuron::PORT_NOT_AVAILABLE )
    {
      nestml_buffer_idx = std::get<0>(rport_to_nestml_buffer_idx[e.get_rport()]);
    }
    weight = -weight;
  }
  B_.spike_inputs_[ nestml_buffer_idx - MIN_SPIKE_RECEPTOR ].add_value(
    e.get_rel_delivery_steps( nest::kernel().simulation_manager.get_slice_origin() ),
    weight * e.get_multiplicity() );
}

