/*
 *  stdp_gaussian_with_separate_exp_r_dependence_connection.h
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

#ifndef STDP_GAUSSIAN_WITH_SEPARATE_EXP_R_DEPENDENCE_CONNECTION_H
#define STDP_GAUSSIAN_WITH_SEPARATE_EXP_R_DEPENDENCE_CONNECTION_H

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

Description
+++++++++++

EndUserDocs */

// connections are templates of target identifier type (used for pointer /
// target index addressing) derived from generic connection template

template < typename targetidentifierT >
class STDPGaussianWithSeparateExpRDependenceConnection : public Connection< targetidentifierT >
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
  STDPGaussianWithSeparateExpRDependenceConnection();


  /**
   * Copy constructor.
   * Needs to be defined properly in order for GenericConnector to work.
   */
  STDPGaussianWithSeparateExpRDependenceConnection( const STDPGaussianWithSeparateExpRDependenceConnection& ) = default;
  STDPGaussianWithSeparateExpRDependenceConnection& operator=( const STDPGaussianWithSeparateExpRDependenceConnection& ) = default;

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
  double A_plus(
    double w,
    double Wmin_,
    double Wmax_,
    double beta_pre
  ) {
    return exp(-beta_pre * (Wmax_ - w) / (Wmax_ - Wmin_));
  }
    
  double A_minus(
    double w,
    double Wmin_,
    double Wmax_,
    double beta_post
  ) {
    return exp(-beta_post * (w - Wmin_) / (Wmax_ - Wmin_));
  }
    
  double x(
    double dt,
    double tNorm,
    double gamma_pre
  ) {
    return 0.5*(1 + std::copysign(1, dt))*fabs(dt/tNorm) * exp(-gamma_pre * pow(dt/tNorm, 2));
  }
    
  double y(
    double dt,
    double tNorm,
    double gamma_post
  ) {
    return 0.5*(1 - std::copysign(1, dt))*fabs(dt/tNorm) * exp(-gamma_post * pow(dt/tNorm, 2));
  }
    
  double global_stdp_func(double dt, double w)
  {
    if (abs(dt) > delta_t_max_)
      // No weight change if the pre- and post-impulses
      // do not intersect.
      return w;
    if (std::isinf(dt))
      // No post-spikes have occurred yet.
      return w;

    w += alpha_plus_ * A_plus(w, Wmin_, Wmax_, beta_plus_) * x(dt, tau_plus_, gamma_plus_)
      - alpha_minus_ * A_minus(w, Wmin_, Wmax_, beta_minus_) * y(dt, tau_minus_, gamma_minus_);

    if (w < Wmin_)
      w = Wmin_;
    if (w > Wmax_)
      w = Wmax_;
    return w;
  }

  double
  facilitate_( double w, double delta_t )
  {
    return global_stdp_func(delta_t, w);
  }

  double
  depress_( double w, double delta_t )
  {
    return global_stdp_func(delta_t, w);
  }

  // data members of each connection
  double weight_;
  double Wmax_;
  double Wmin_;
  double t_lastspike_;
  double delta_t_max_;
  double alpha_plus_, alpha_minus_, beta_plus_, beta_minus_, gamma_plus_, gamma_minus_, tau_plus_, tau_minus_;
};

template < typename targetidentifierT >
constexpr ConnectionModelProperties STDPGaussianWithSeparateExpRDependenceConnection< targetidentifierT >::properties;

/**
 * Send an event to the receiver of this connection.
 * \param e The event to send
 * \param t The thread on which this connection is stored.
 * \param cp Common properties object, containing the stdp parameters.
 */
template < typename targetidentifierT >
inline void
STDPGaussianWithSeparateExpRDependenceConnection< targetidentifierT >::send( Event& e, size_t t, const CommonSynapseProperties& )
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
    // an array overflow in Archiving node.
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
STDPGaussianWithSeparateExpRDependenceConnection< targetidentifierT >::STDPGaussianWithSeparateExpRDependenceConnection()
  : ConnectionBase()
  , weight_( 1. )
  , Wmax_( 1. )
  , Wmin_( 0. )
  , t_lastspike_( -INFINITY )
  , delta_t_max_( 500. )
  , alpha_plus_( 0.3161977426412055 )
  , alpha_minus_( 0.011297914920262086 )
  , beta_plus_( 2.213043124548639 )
  , beta_minus_( -5.9693974750170025 )
  , gamma_plus_( 0.031839429846118886 )
  , gamma_minus_( 0.14568323501051358 )
  , tau_plus_( 10 )
  , tau_minus_( 10 )
{
}

namespace mynames_stdp_gaussian_with_separate_exp_r_dependence_connection{
  const Name alpha_plus( "alpha_plus" );
  const Name alpha_minus( "alpha_minus" );
  const Name beta_plus( "beta_plus" );
  const Name beta_minus( "beta_minus" );
  const Name gamma_plus( "gamma_plus" );
  const Name gamma_minus( "gamma_minus" );
  const Name tau_plus( "tau_plus" );
  const Name tau_minus( "tau_minus" );
}

template < typename targetidentifierT >
void
STDPGaussianWithSeparateExpRDependenceConnection< targetidentifierT >::get_status( DictionaryDatum& d ) const
{
  ConnectionBase::get_status( d );
  def< double >( d, names::weight, weight_ );
  def< double >( d, names::Wmax, Wmax_ );
  def< double >( d, names::Wmin, Wmin_ );
  def< double >( d, names::delta_t_max, delta_t_max_ );
  def< double >( d, mynames_stdp_gaussian_with_separate_exp_r_dependence_connection::alpha_plus, alpha_plus_ );
  def< double >( d, mynames_stdp_gaussian_with_separate_exp_r_dependence_connection::alpha_minus, alpha_minus_ );
  def< double >( d, mynames_stdp_gaussian_with_separate_exp_r_dependence_connection::beta_plus, beta_plus_ );
  def< double >( d, mynames_stdp_gaussian_with_separate_exp_r_dependence_connection::beta_minus, beta_minus_ );
  def< double >( d, mynames_stdp_gaussian_with_separate_exp_r_dependence_connection::gamma_plus, gamma_plus_ );
  def< double >( d, mynames_stdp_gaussian_with_separate_exp_r_dependence_connection::gamma_minus, gamma_minus_ );
  def< double >( d, mynames_stdp_gaussian_with_separate_exp_r_dependence_connection::tau_plus, tau_plus_ );
  def< double >( d, mynames_stdp_gaussian_with_separate_exp_r_dependence_connection::tau_minus, tau_minus_ );
  def< long >( d, names::size_of, sizeof( *this ) );
}

template < typename targetidentifierT >
void
STDPGaussianWithSeparateExpRDependenceConnection< targetidentifierT >::set_status( const DictionaryDatum& d, ConnectorModel& cm )
{
  ConnectionBase::set_status( d, cm );
  updateValue< double >( d, names::weight, weight_ );
  updateValue< double >( d, names::weight, weight_ );
  updateValue< double >( d, names::Wmax, Wmax_ );
  updateValue< double >( d, names::Wmin, Wmin_ );
  updateValue< double >( d, names::delta_t_max, delta_t_max_ );
  updateValue< double >( d, mynames_stdp_gaussian_with_separate_exp_r_dependence_connection::alpha_plus, alpha_plus_ );
  updateValue< double >( d, mynames_stdp_gaussian_with_separate_exp_r_dependence_connection::alpha_minus, alpha_minus_ );
  updateValue< double >( d, mynames_stdp_gaussian_with_separate_exp_r_dependence_connection::beta_plus, beta_plus_ );
  updateValue< double >( d, mynames_stdp_gaussian_with_separate_exp_r_dependence_connection::beta_minus, beta_minus_ );
  updateValue< double >( d, mynames_stdp_gaussian_with_separate_exp_r_dependence_connection::gamma_plus, gamma_plus_ );
  updateValue< double >( d, mynames_stdp_gaussian_with_separate_exp_r_dependence_connection::gamma_minus, gamma_minus_ );
  updateValue< double >( d, mynames_stdp_gaussian_with_separate_exp_r_dependence_connection::tau_plus, tau_plus_ );
  updateValue< double >( d, mynames_stdp_gaussian_with_separate_exp_r_dependence_connection::tau_minus, tau_minus_ );

  if ( Wmin_ > Wmax_ )
  {
    throw BadProperty( "Wmin should be lower than Wmax" );
  }
}

} // of namespace nest

#endif // of #ifndef STDP_GAUSSIAN_WITH_SEPARATE_EXP_R_DEPENDENCE_CONNECTION_H
