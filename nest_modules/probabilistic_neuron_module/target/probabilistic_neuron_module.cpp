
/*
*  probabilistic_neuron_module.cpp
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
*  2024-07-09 06:06:00.178313
*/

// Include from NEST
#include "nest_extension_interface.h"

// include headers with your own stuff


#include "probabilistic_neuron.h"



class probabilistic_neuron_module : public nest::NESTExtensionInterface
{
  public:
    probabilistic_neuron_module() {}
    ~probabilistic_neuron_module() {}

    void initialize() override;
};

probabilistic_neuron_module probabilistic_neuron_module_LTX_module;

void probabilistic_neuron_module::initialize()
{
    // register neurons
    register_probabilistic_neuron("probabilistic_neuron");
}
