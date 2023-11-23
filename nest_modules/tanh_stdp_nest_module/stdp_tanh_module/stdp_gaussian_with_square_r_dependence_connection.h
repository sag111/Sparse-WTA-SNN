/*
 *  stdp_gaussian_with_square_r_dependence_connection.h
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

#ifndef STDP_GAUSSIAN_WITH_SQUARE_R_DEPENDENCE_CONNECTION_H
#define STDP_GAUSSIAN_WITH_SQUARE_R_DEPENDENCE_CONNECTION_H

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

The fit for the memristors measured in June 2021.

Description
+++++++++++

delta_G(delta_t, R_initial) = F(delta_t),
where
F(x) = (ax^2 + bx + c) * exp(-|d|(x-f)^2)
and each of a, b, c, d depends on R_initial as follows:
p1 * R^2 + p2 * R + p3, p in {a,b,c,d,f}

EndUserDocs */

// connections are templates of target identifier type (used for pointer /
// target index addressing) derived from generic connection template

template < typename targetidentifierT >
class STDPGaussianWithSquareRDependenceConnection : public Connection< targetidentifierT >
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
  STDPGaussianWithSquareRDependenceConnection();


  /**
   * Copy constructor.
   * Needs to be defined properly in order for GenericConnector to work.
   */
  STDPGaussianWithSquareRDependenceConnection( const STDPGaussianWithSquareRDependenceConnection& ) = default;
  STDPGaussianWithSquareRDependenceConnection& operator=( const STDPGaussianWithSquareRDependenceConnection& ) = default;

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
  double
  parameter_func(double x, double a, double b, double c)
  {
    return a * x * x + b * x + c;
  }

  double
  stdp_func_for_current_R(double x, double a, double b, double c, double d, double f)
  {
    return (a * x * x + b * x + c) * exp(
      - abs(d) * ((x + f) * (x + f))
    );
  }

  double global_stdp_func( double w, double delta_t )
  {
    double R = (1 / w);
    #define R_max (1 / Wmax_)
    
    R = R / R_max;
    delta_t = delta_t / delta_t_max_;

    if (abs(delta_t) > 1)
      // No weight change if the pre- and post-impulses
      // do not intersect.
      return w;
    if (std::isinf(delta_t))
      // No post-spikes have occurred yet.
      return w;

    double a = parameter_func(R, a1_, a2_, a3_);
    double b = parameter_func(R, b1_, b2_, b3_);
    double c = parameter_func(R, c1_, c2_, c3_);
    double d = parameter_func(R, d1_, d2_, d3_);
    double f = parameter_func(R, f1_, f2_, f3_);

    w += w * 0.01*stdp_func_for_current_R(delta_t, a, b, c, d, f);

    if (w < Wmin_)
      w = Wmin_;
    if (w > Wmax_)
      w = Wmax_;
    return w;
  }

  double
  facilitate_( double w, double delta_t )
  {
    return global_stdp_func(w, delta_t);
  }

  double
  depress_( double w, double delta_t )
  {
    return global_stdp_func(w, delta_t);
  }

  // data members of each connection
  double weight_;
  double Wmax_;
  double Wmin_;
  double t_lastspike_;
  double delta_t_max_;
  double a1_, a2_, a3_;
  double b1_, b2_, b3_;
  double c1_, c2_, c3_;
  double d1_, d2_, d3_;
  double f1_, f2_, f3_;
};

template < typename targetidentifierT >
constexpr ConnectionModelProperties STDPGaussianWithSquareRDependenceConnection< targetidentifierT >::properties;

/**
 * Send an event to the receiver of this connection.
 * \param e The event to send
 * \param t The thread on which this connection is stored.
 * \param cp Common properties object, containing the stdp parameters.
 */
template < typename targetidentifierT >
inline void
STDPGaussianWithSquareRDependenceConnection< targetidentifierT >::send( Event& e, size_t t, const CommonSynapseProperties& )
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
STDPGaussianWithSquareRDependenceConnection< targetidentifierT >::STDPGaussianWithSquareRDependenceConnection()
  : ConnectionBase()
  , weight_( 1.0 )
  , Wmax_( 1.0 / 0.1 ) // kOhm
  , Wmin_( 1.0 / 100 ) // kOhm
  , t_lastspike_( -INFINITY )
  , delta_t_max_( 5e+2 )
  , a1_( -26913.86853978 ), a2_(21827.06401163), a3_(-4494.70432941)
  , b1_( -28271.01897556 ), b2_(22377.70719166), b3_(-3820.53849135)
  , c1_( 6986.56517422 ), c2_(-4096.087957), c3_(826.57827891)
  , d1_( -12796.20679118 ), d2_(14748.62176877), d3_(-2820.15647475)
  , f1_( -0.09180867 ), f2_(0.07801423), f3_(-0.10783657)
{
}

namespace names
{
  const Name Wmin ( "Wmin" );
  const Name delta_t_max ( "delta_t_max" );
  const Name a1( "a1" ), a2( "a2" ), a3( "a3" );
  const Name b1( "b1" ), b2( "b2" ), b3( "b3" );
  const Name c1( "c1" ), c2( "c2" ), c3( "c3" );
  const Name d1( "d1" ), d2( "d2" ), d3( "d3" );
  const Name f1( "f1" ), f2( "f2" ), f3( "f3" );
}

template < typename targetidentifierT >
void
STDPGaussianWithSquareRDependenceConnection< targetidentifierT >::get_status( DictionaryDatum& d ) const
{
  ConnectionBase::get_status( d );
  def< double >( d, names::weight, weight_ );
  def< double >( d, names::Wmax, Wmax_ );
  def< double >( d, names::Wmin, Wmin_ );
  def< double >( d, names::delta_t_max, delta_t_max_ );
  def< double >( d, names::a1, a1_ );
  def< double >( d, names::a2, a2_ );
  def< double >( d, names::a3, a3_ );
  def< double >( d, names::b1, b1_ );
  def< double >( d, names::b2, b2_ );
  def< double >( d, names::b3, b3_ );
  def< double >( d, names::c1, c1_ );
  def< double >( d, names::c2, c2_ );
  def< double >( d, names::c3, c3_ );
  def< double >( d, names::d1, d1_ );
  def< double >( d, names::d2, d2_ );
  def< double >( d, names::d3, d3_ );
  def< double >( d, names::f1, f1_ );
  def< double >( d, names::f2, f2_ );
  def< double >( d, names::f3, f3_ );
  def< long >( d, names::size_of, sizeof( *this ) );
}

template < typename targetidentifierT >
void
STDPGaussianWithSquareRDependenceConnection< targetidentifierT >::set_status( const DictionaryDatum& d, ConnectorModel& cm )
{
  ConnectionBase::set_status( d, cm );
  updateValue< double >( d, names::weight, weight_ );
  updateValue< double >( d, names::Wmax, Wmax_ );
  updateValue< double >( d, names::Wmin, Wmin_ );
  updateValue< double >( d, names::delta_t_max, delta_t_max_ );
  updateValue< double >( d, names::a1, a1_ );
  updateValue< double >( d, names::a2, a2_ );
  updateValue< double >( d, names::a3, a3_ );
  updateValue< double >( d, names::b1, b1_ );
  updateValue< double >( d, names::b2, b2_ );
  updateValue< double >( d, names::b3, b3_ );
  updateValue< double >( d, names::c1, c1_ );
  updateValue< double >( d, names::c2, c2_ );
  updateValue< double >( d, names::c3, c3_ );
  updateValue< double >( d, names::d1, d1_ );
  updateValue< double >( d, names::d2, d2_ );
  updateValue< double >( d, names::d3, d3_ );
  updateValue< double >( d, names::f1, f1_ );
  updateValue< double >( d, names::f2, f2_ );
  updateValue< double >( d, names::f3, f3_ );

  if ( Wmin_ > Wmax_ )
  {
    throw BadProperty( "Wmin should be lower than Wmax" );
  }
}

} // of namespace nest

#endif // of #ifndef STDP_GAUSSIAN_WITH_SQUARE_R_DEPENDENCE_CONNECTION_H
