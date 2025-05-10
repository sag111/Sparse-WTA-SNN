/*
 *  stdp_tanh_connection.h
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
 */

#ifndef STDP_TANH_CONNECTION_H
#define STDP_TANH_CONNECTION_H

// C++ includes:
#include <cmath>

// Includes from nestkernel:
#include "common_synapse_properties.h"
#include "connection.h"
#include "connector_model.h"
#include "event.h"

// Includes from sli:
#include "dictdatum.h"
#include "dictutils.h"

namespace nest
{

/* BeginUserDocs: synapse, spike-timing-dependent plasticity

Short description
+++++++++++++++++

stdp_tanh_synapse is a connector to create synapses with
Nekhaev's fit [1] of memristive STDP:

Description
+++++++++++

  w := w + (
    facilitate(delta_t, w/Wmax) if delta_t > 0 or
    depress(delta_t, w/Wmax) if delta_t < 0
  ),
where
  facilitate(delta_t, w) =
    a_plus * (np.tanh(-(delta_t-mu_plus)/tau_plus)+0)
    + c_plus*delta_t + a_plus*w * w_d_plus
  depress(delta_t, w/Wmax) =
    -a_minus * (np.tanh((delta_t-mu_minus)/tau_minus)+0)
    + c_minus*delta_t + a_minus*w * w_d_minus


The spike pairing scheme is symmetric nearest-neighbour.
The default constants are from [1], except that w_d is 0.

Parameters
++++++++++

(for each name below, two parameters exist,
with suffix _plus for facilitation equation and
with suffix _minus for depression equation)

===== ======  ================================
 a    real    Weight change amplitude
 mu   ms      Offset subtracted from delta_t
              when passed to tanh
 tau  ms      Denominator for delta_t
              when passed to tanh
 c    1/ms    the coefficient of the term
              that depends linearly on delta_t
 w_d  real    constant part of weight change
 Wmax real    Maximum allowed weight
===== ======  ================================

Transmits
+++++++++

SpikeEvent

References
++++++++++

   [1] Demin, Vyacheslav & Nekhaev, Dmitry & Surazhevsky, I.A. &
       Nikiruy, Kristina & Emelyanov, Andrey & Nikolaev, Sergey &
       Rylkov, V. & Kovalchuk, M.V.. (2021). Necessary conditions for
       STDP-based pattern recognition learning in a memristive spiking
       neural network. Neural Networks. 134. 64-75.
       10.1016/j.neunet.2020.11.005. 

See also
++++++++

stdp_nn_symm_synapse

EndUserDocs */

// connections are templates of target identifier type (used for pointer /
// target index addressing) derived from generic connection template

template < typename targetidentifierT >
class STDPtanhConnection : public Connection< targetidentifierT >
{

public:
  typedef CommonSynapseProperties CommonPropertiesType;
  typedef Connection< targetidentifierT > ConnectionBase;

  static constexpr ConnectionModelProperties properties = ConnectionModelProperties::HAS_DELAY  
    | ConnectionModelProperties::IS_PRIMARY | ConnectionModelProperties::SUPPORTS_HPC
    | ConnectionModelProperties::SUPPORTS_LBL;

  /**
   * Default Constructor.
   * Sets default values for all parameters. Needed by GenericConnectorModel.
   */
  STDPtanhConnection();


  /**
   * Copy constructor.
   * Needs to be defined properly in order for GenericConnector to work.
   */
  STDPtanhConnection( const STDPtanhConnection& ) = default;
  STDPtanhConnection& operator=( const STDPtanhConnection& ) = default;

  // Explicitly declare all methods inherited from the dependent base
  // ConnectionBase. This avoids explicit name prefixes in all places these
  // functions are used. Since ConnectionBase depends on the template parameter,
  // they are not automatically found in the base class.
  using ConnectionBase::get_delay;
  using ConnectionBase::get_delay_steps;
  using ConnectionBase::get_rport;
  using ConnectionBase::get_target;

  /**
   * Get all properties of this connection and put them into a dictionary.
   */
  void get_status( DictionaryDatum& d ) const;

  /**
   * Set properties of this connection from the values given in dictionary.
   */
  void set_status( const DictionaryDatum& d, ConnectorModel& cm );

  /**
   * Send an event to the receiver of this connection.
   * \param e The event to send
   * \param cp common properties of all synapses (empty).
   */
  void send( Event& e, size_t t, const CommonSynapseProperties& cp );


  class ConnTestDummyNode : public ConnTestDummyNodeBase
  {
  public:
    // Ensure proper overriding of overloaded virtual functions.
    // Return values from functions are ignored.
    using ConnTestDummyNodeBase::handles_test_event;
    size_t
    handles_test_event( SpikeEvent&, size_t ) override
    {
      return invalid_port;
    }
  };

  void
  check_connection( Node& s, Node& t, size_t receptor_type, const CommonPropertiesType& )
  {
    ConnTestDummyNode dummy_target;

    ConnectionBase::check_connection_( dummy_target, s, t, receptor_type );

    t.register_stdp_connection( t_lastspike_ - get_delay(), get_delay() );
  }

  void
  set_weight( double w )
  {
    weight_ = w;
  }

private:
  double func_pos( double x, double a, double mu, double tau, double c, double d )
  {
    return a * (tanh(-(x-mu)/tau)+0) + c*x + d;
  }

  double func_neg( double x, double a, double mu, double tau, double c, double d )
  {
    return -a * (tanh((x-mu)/tau)+0) + c*x + d;
  }

  double get_dw_pos(double delta_t, double w )
  {
    return (
      // Check for infinity in order to handle the case
      // when the first post-spike has no pre- to pair with.
      // The constant part of weight change should still appply,
      // which is why t_lastspike_ is initialized with infinity
      // and explicitly handled here.
      std::isinf(delta_t) ? 0 : func_pos(
        delta_t, a_plus_ * w, mu_plus_, tau_plus_, c_plus_, a_plus_ * w
      )
      + w_d_plus_
    );
  }

  double get_dw_neg(double delta_t, double w )
  {
    return (
      std::isinf(delta_t) ? 0 : func_neg(
        delta_t, a_minus_ * w, mu_minus_, tau_minus_, c_minus_, a_minus_ * w
      )
      + w_d_minus_
    );
  }


  double
  facilitate_( double w, double delta_t )
  {
    double norm_w = w / Wmax_;
    norm_w += get_dw_pos(delta_t, norm_w);
    return norm_w < 1.0 ? norm_w * Wmax_ : Wmax_;
  }

  double
  depress_( double w, double delta_t )
  {
    double norm_w = w / Wmax_;
    norm_w += get_dw_neg(delta_t, norm_w);
    return norm_w > 0.0 ? norm_w * Wmax_ : 0.0;
  }

  // data members of each connection
  double weight_;
  double Wmax_;
  double a_plus_;
  double a_minus_;
  double mu_plus_;
  double mu_minus_;
  double tau_plus_;
  double tau_minus_;
  double c_plus_;
  double c_minus_;
  double w_d_plus_;
  double w_d_minus_;
  double t_lastspike_;
};

template < typename targetidentifierT >
constexpr ConnectionModelProperties STDPtanhConnection< targetidentifierT >::properties;

/**
 * Send an event to the receiver of this connection.
 * \param e The event to send
 * \param t The thread on which this connection is stored.
 * \param cp Common properties object, containing the stdp parameters.
 */
template < typename targetidentifierT >
inline void
STDPtanhConnection< targetidentifierT >::send( Event& e, size_t t, const CommonSynapseProperties& )
{
  // synapse STDP depressing/facilitation dynamics
  const double t_spike = e.get_stamp().get_ms();

  // use accessor functions (inherited from Connection< >) to obtain delay and
  // target
  Node* target = get_target( t );
  double dendritic_delay = get_delay();

  // get spike history in relevant range (t1, t2] from postsynaptic neuron
  std::deque< histentry >::iterator start;
  std::deque< histentry >::iterator finish;

  // For a new synapse, t_lastspike_ contains the point in time of the last
  // spike. So we initially read the
  // history(t_last_spike - dendritic_delay, ..., T_spike-dendritic_delay]
  // which increases the access counter for these entries.
  // At registration, all entries' access counters of
  // history[0, ..., t_last_spike - dendritic_delay] have been
  // incremented by ArchivingNode::register_stdp_connection(). See bug #218 for
  // details.
  target->get_history( t_lastspike_ - dendritic_delay, t_spike - dendritic_delay, &start, &finish );

  // facilitation due to post-synaptic spikes t_
  // since the previous pre-synaptic spike t_lastspike_
  double delta_t;
  while ( start != finish )
  {
    delta_t = ( start->t_ + dendritic_delay ) - t_lastspike_;
    ++start;

    // get_history() should make sure that
    // start->t_ > t_lastspike_ - dendritic_delay, i.e. delta_t > 0
    assert( delta_t > kernel().connection_manager.get_stdp_eps() );

    weight_ = facilitate_( weight_, delta_t );
  }

  // depression due to the latest post-synaptic spike finish->t_
  // before the current pre-synaptic spike t_spike.
  if ( start == finish )
  {
    // Request the full spike history.
    // While inefficient, it ensures that any pre-spike causes
    // a weight update.
    // Presumably, we can pass -1 as starting time, this should not cause
    // an array overflow in Acrchving node.
    target->get_history( -1, t_spike - dendritic_delay, &start, &finish );
  }
  if ( start != finish )
  {
    --finish;
    // Now we will have delta_t < 0,
    // but it will still be post minus pre.
    delta_t = ( finish->t_ + dendritic_delay ) - t_spike;
    weight_ = depress_( weight_, delta_t );
  }

  e.set_receiver( *target );
  e.set_weight( weight_ );
  // use accessor functions (inherited from Connection< >) to obtain delay in
  // steps and rport
  e.set_delay_steps( get_delay_steps() );
  e.set_rport( get_rport() );
  e();

  t_lastspike_ = t_spike;
}


template < typename targetidentifierT >
STDPtanhConnection< targetidentifierT >::STDPtanhConnection()
  : ConnectionBase()
  , weight_( 1.0 )
  , Wmax_( 1.0 )
  , a_plus_( 0.07376001452742792 )
  , a_minus_( -0.04963699617346189 )
  , mu_plus_( 26.730656991521247 )
  , mu_minus_( -22.325265872889496 )
  , tau_plus_( 9.300596954566565 )
  , tau_minus_( -10.81180238558187 )
  , c_plus_( 0.0 )
  , c_minus_( 0.0 )
  , w_d_plus_( 0.0 )
  , w_d_minus_( 0.0 )
  , t_lastspike_( -INFINITY )
{
}

namespace names
{
  const Name a_plus ( "a_plus" );
  const Name a_minus ( "a_minus" );
  const Name mu_plus ( "mu_plus" );
  const Name mu_minus ( "mu_minus" );
  const Name tau_plus ( "tau_plus" );
  const Name tau_minus ( "tau_minus" );
  const Name c_plus ( "c_plus" );
  const Name c_minus ( "c_minus" );
  const Name w_d_plus ( "w_d_plus" );
  const Name w_d_minus ( "w_d_minus" );
}

template < typename targetidentifierT >
void
STDPtanhConnection< targetidentifierT >::get_status( DictionaryDatum& d ) const
{
  ConnectionBase::get_status( d );
  def< double >( d, names::weight, weight_ );
  def< double >( d, names::a_plus, a_plus_ );
  def< double >( d, names::a_minus, a_minus_ );
  def< double >( d, names::mu_plus, mu_plus_ );
  def< double >( d, names::mu_minus, mu_minus_ );
  def< double >( d, names::tau_plus, tau_plus_ );
  def< double >( d, names::tau_minus, tau_minus_ );
  def< double >( d, names::c_plus, c_plus_ );
  def< double >( d, names::c_minus, c_minus_ );
  def< double >( d, names::w_d_plus, w_d_plus_ );
  def< double >( d, names::w_d_minus, w_d_minus_ );
  def< double >( d, names::Wmax, Wmax_ );
  def< long >( d, names::size_of, sizeof( *this ) );
}

template < typename targetidentifierT >
void
STDPtanhConnection< targetidentifierT >::set_status( const DictionaryDatum& d, ConnectorModel& cm )
{
  ConnectionBase::set_status( d, cm );
  updateValue< double >( d, names::weight, weight_ );
  updateValue< double >( d, names::a_plus, a_plus_ );
  updateValue< double >( d, names::a_minus, a_minus_ );
  updateValue< double >( d, names::mu_plus, mu_plus_ );
  updateValue< double >( d, names::mu_minus, mu_minus_ );
  updateValue< double >( d, names::tau_plus, tau_plus_ );
  updateValue< double >( d, names::tau_minus, tau_minus_ );
  updateValue< double >( d, names::c_plus, c_plus_ );
  updateValue< double >( d, names::c_minus, c_minus_ );
  updateValue< double >( d, names::w_d_plus, w_d_plus_ );
  updateValue< double >( d, names::w_d_minus, w_d_minus_ );
  updateValue< double >( d, names::Wmax, Wmax_ );

  // check if weight_ and Wmax_ has the same sign
  if ( not( ( ( weight_ >= 0 ) - ( weight_ < 0 ) ) == ( ( Wmax_ >= 0 ) - ( Wmax_ < 0 ) ) ) )
  {
    throw BadProperty( "Weight and Wmax must have same sign." );
  }
}

} // of namespace nest

#endif // of #ifndef STDP_TANH_CONNECTION_H
