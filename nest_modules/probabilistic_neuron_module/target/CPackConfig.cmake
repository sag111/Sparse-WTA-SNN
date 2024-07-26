# This file will be configured to contain variables for CPack. These variables
# should be set in the CMake list file of the project before CPack module is
# included. The list of available CPACK_xxx variables and their associated
# documentation may be obtained using
#  cpack --help-variable-list
#
# Some variables are common to all generators (e.g. CPACK_PACKAGE_NAME)
# and some are specific to a generator
# (e.g. CPACK_NSIS_EXTRA_INSTALL_COMMANDS). The generator specific variables
# usually begin with CPACK_<GENNAME>_xxxx.


set(CPACK_BUILD_SOURCE_DIRS "/mnt/c/Users/AI_Engineer_Yury/Projects/Sparse-WTA-SNN/nest_modules/probabilistic_neuron_module/target;/mnt/c/Users/AI_Engineer_Yury/Projects/Sparse-WTA-SNN/nest_modules/probabilistic_neuron_module/target")
set(CPACK_CMAKE_GENERATOR "Unix Makefiles")
set(CPACK_COMPONENTS_ALL "")
set(CPACK_COMPONENT_UNSPECIFIED_HIDDEN "TRUE")
set(CPACK_COMPONENT_UNSPECIFIED_REQUIRED "TRUE")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_FILE "/home/yurydavydov/miniconda3/envs/nestenv/share/cmake-3.30/Templates/CPack.GenericDescription.txt")
set(CPACK_DEFAULT_PACKAGE_DESCRIPTION_SUMMARY "probabilistic_neuron_module built using CMake")
set(CPACK_GENERATOR "TGZ")
set(CPACK_INNOSETUP_ARCHITECTURE "x64")
set(CPACK_INSTALL_CMAKE_PROJECTS "/mnt/c/Users/AI_Engineer_Yury/Projects/Sparse-WTA-SNN/nest_modules/probabilistic_neuron_module/target;probabilistic_neuron_module;ALL;/")
set(CPACK_INSTALL_PREFIX "/home/yurydavydov/miniconda3/envs/nestenv")
set(CPACK_MODULE_PATH "")
set(CPACK_NSIS_DISPLAY_NAME "probabilistic_neuron_module 1.0")
set(CPACK_NSIS_INSTALLER_ICON_CODE "")
set(CPACK_NSIS_INSTALLER_MUI_ICON_CODE "")
set(CPACK_NSIS_INSTALL_ROOT "$PROGRAMFILES")
set(CPACK_NSIS_PACKAGE_NAME "probabilistic_neuron_module 1.0")
set(CPACK_NSIS_UNINSTALL_NAME "Uninstall")
set(CPACK_OBJCOPY_EXECUTABLE "/home/yurydavydov/miniconda3/envs/nestenv/bin/objcopy")
set(CPACK_OBJDUMP_EXECUTABLE "/home/yurydavydov/miniconda3/envs/nestenv/bin/objdump")
set(CPACK_OUTPUT_CONFIG_FILE "/mnt/c/Users/AI_Engineer_Yury/Projects/Sparse-WTA-SNN/nest_modules/probabilistic_neuron_module/target/CPackConfig.cmake")
set(CPACK_PACKAGE_DEFAULT_LOCATION "/")
set(CPACK_PACKAGE_DESCRIPTION_FILE "/home/yurydavydov/miniconda3/envs/nestenv/share/cmake-3.30/Templates/CPack.GenericDescription.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "NEST Module probabilistic_neuron_module")
set(CPACK_PACKAGE_FILE_NAME "probabilistic_neuron_module-1.0-Linux")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "probabilistic_neuron_module 1.0")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "probabilistic_neuron_module 1.0")
set(CPACK_PACKAGE_NAME "probabilistic_neuron_module")
set(CPACK_PACKAGE_RELOCATABLE "true")
set(CPACK_PACKAGE_VENDOR "NEST Initiative (http://www.nest-initiative.org/)")
set(CPACK_PACKAGE_VERSION "1.0")
set(CPACK_PACKAGE_VERSION_MAJOR "1")
set(CPACK_PACKAGE_VERSION_MINOR "0")
set(CPACK_PACKAGE_VERSION_PATCH "1")
set(CPACK_READELF_EXECUTABLE "/home/yurydavydov/miniconda3/envs/nestenv/bin/readelf")
set(CPACK_RESOURCE_FILE_LICENSE "/home/yurydavydov/miniconda3/envs/nestenv/share/cmake-3.30/Templates/CPack.GenericLicense.txt")
set(CPACK_RESOURCE_FILE_README "/home/yurydavydov/miniconda3/envs/nestenv/share/cmake-3.30/Templates/CPack.GenericDescription.txt")
set(CPACK_RESOURCE_FILE_WELCOME "/home/yurydavydov/miniconda3/envs/nestenv/share/cmake-3.30/Templates/CPack.GenericWelcome.txt")
set(CPACK_SET_DESTDIR "OFF")
set(CPACK_SOURCE_GENERATOR "TGZ")
set(CPACK_SOURCE_IGNORE_FILES "\\.gitignore;\\.git/;\\.travis\\.yml;/build/;/_CPack_Packages/;CMakeFiles/;cmake_install\\.cmake;Makefile.*;CMakeCache\\.txt;CPackConfig\\.cmake;CPackSourceConfig\\.cmake")
set(CPACK_SOURCE_OUTPUT_CONFIG_FILE "/mnt/c/Users/AI_Engineer_Yury/Projects/Sparse-WTA-SNN/nest_modules/probabilistic_neuron_module/target/CPackSourceConfig.cmake")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "probabilistic_neuron_module")
set(CPACK_SYSTEM_NAME "Linux")
set(CPACK_THREADS "1")
set(CPACK_TOPLEVEL_TAG "Linux")
set(CPACK_WIX_SIZEOF_VOID_P "8")

if(NOT CPACK_PROPERTIES_FILE)
  set(CPACK_PROPERTIES_FILE "/mnt/c/Users/AI_Engineer_Yury/Projects/Sparse-WTA-SNN/nest_modules/probabilistic_neuron_module/target/CPackProperties.cmake")
endif()

if(EXISTS ${CPACK_PROPERTIES_FILE})
  include(${CPACK_PROPERTIES_FILE})
endif()
