# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.30

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/Cellar/cmake/3.30.5/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/etienne/Documents/Developer/Iterative_methods

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/etienne/Documents/Developer/Iterative_methods/build

# Include any dependencies generated for this target.
include CMakeFiles/2_jacobi_weak_scalability.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/2_jacobi_weak_scalability.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/2_jacobi_weak_scalability.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/2_jacobi_weak_scalability.dir/flags.make

CMakeFiles/2_jacobi_weak_scalability.dir/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp.o: CMakeFiles/2_jacobi_weak_scalability.dir/flags.make
CMakeFiles/2_jacobi_weak_scalability.dir/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp.o: /Users/etienne/Documents/Developer/Iterative_methods/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp
CMakeFiles/2_jacobi_weak_scalability.dir/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp.o: CMakeFiles/2_jacobi_weak_scalability.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/etienne/Documents/Developer/Iterative_methods/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/2_jacobi_weak_scalability.dir/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/2_jacobi_weak_scalability.dir/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp.o -MF CMakeFiles/2_jacobi_weak_scalability.dir/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp.o.d -o CMakeFiles/2_jacobi_weak_scalability.dir/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp.o -c /Users/etienne/Documents/Developer/Iterative_methods/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp

CMakeFiles/2_jacobi_weak_scalability.dir/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/2_jacobi_weak_scalability.dir/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/etienne/Documents/Developer/Iterative_methods/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp > CMakeFiles/2_jacobi_weak_scalability.dir/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp.i

CMakeFiles/2_jacobi_weak_scalability.dir/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/2_jacobi_weak_scalability.dir/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/etienne/Documents/Developer/Iterative_methods/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp -o CMakeFiles/2_jacobi_weak_scalability.dir/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp.s

CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_mpi.cpp.o: CMakeFiles/2_jacobi_weak_scalability.dir/flags.make
CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_mpi.cpp.o: /Users/etienne/Documents/Developer/Iterative_methods/solvers/jacobi_mpi.cpp
CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_mpi.cpp.o: CMakeFiles/2_jacobi_weak_scalability.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/etienne/Documents/Developer/Iterative_methods/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_mpi.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_mpi.cpp.o -MF CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_mpi.cpp.o.d -o CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_mpi.cpp.o -c /Users/etienne/Documents/Developer/Iterative_methods/solvers/jacobi_mpi.cpp

CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_mpi.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_mpi.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/etienne/Documents/Developer/Iterative_methods/solvers/jacobi_mpi.cpp > CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_mpi.cpp.i

CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_mpi.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_mpi.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/etienne/Documents/Developer/Iterative_methods/solvers/jacobi_mpi.cpp -o CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_mpi.cpp.s

CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_sequential.cpp.o: CMakeFiles/2_jacobi_weak_scalability.dir/flags.make
CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_sequential.cpp.o: /Users/etienne/Documents/Developer/Iterative_methods/solvers/jacobi_sequential.cpp
CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_sequential.cpp.o: CMakeFiles/2_jacobi_weak_scalability.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/etienne/Documents/Developer/Iterative_methods/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_sequential.cpp.o"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_sequential.cpp.o -MF CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_sequential.cpp.o.d -o CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_sequential.cpp.o -c /Users/etienne/Documents/Developer/Iterative_methods/solvers/jacobi_sequential.cpp

CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_sequential.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_sequential.cpp.i"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/etienne/Documents/Developer/Iterative_methods/solvers/jacobi_sequential.cpp > CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_sequential.cpp.i

CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_sequential.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_sequential.cpp.s"
	/Library/Developer/CommandLineTools/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/etienne/Documents/Developer/Iterative_methods/solvers/jacobi_sequential.cpp -o CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_sequential.cpp.s

# Object files for target 2_jacobi_weak_scalability
2_jacobi_weak_scalability_OBJECTS = \
"CMakeFiles/2_jacobi_weak_scalability.dir/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp.o" \
"CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_mpi.cpp.o" \
"CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_sequential.cpp.o"

# External object files for target 2_jacobi_weak_scalability
2_jacobi_weak_scalability_EXTERNAL_OBJECTS =

2_jacobi_weak_scalability: CMakeFiles/2_jacobi_weak_scalability.dir/study_cases/2_jacobi_performances/2_jacobi_weak_scalability.cpp.o
2_jacobi_weak_scalability: CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_mpi.cpp.o
2_jacobi_weak_scalability: CMakeFiles/2_jacobi_weak_scalability.dir/solvers/jacobi_sequential.cpp.o
2_jacobi_weak_scalability: CMakeFiles/2_jacobi_weak_scalability.dir/build.make
2_jacobi_weak_scalability: /opt/homebrew/Cellar/open-mpi/5.0.5/lib/libmpi.dylib
2_jacobi_weak_scalability: CMakeFiles/2_jacobi_weak_scalability.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/etienne/Documents/Developer/Iterative_methods/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable 2_jacobi_weak_scalability"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/2_jacobi_weak_scalability.dir/link.txt --verbose=$(VERBOSE)
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold "Giving execute permission to the jacobi_weak_scalability_srcs executable"
	/opt/homebrew/Cellar/cmake/3.30.5/bin/cmake -E echo Giving\ execute\ permission\ to\ jacobi_weak_scalability_srcs
	chmod +x /Users/etienne/Documents/Developer/Iterative_methods/build/2_jacobi_weak_scalability

# Rule to build all files generated by this target.
CMakeFiles/2_jacobi_weak_scalability.dir/build: 2_jacobi_weak_scalability
.PHONY : CMakeFiles/2_jacobi_weak_scalability.dir/build

CMakeFiles/2_jacobi_weak_scalability.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/2_jacobi_weak_scalability.dir/cmake_clean.cmake
.PHONY : CMakeFiles/2_jacobi_weak_scalability.dir/clean

CMakeFiles/2_jacobi_weak_scalability.dir/depend:
	cd /Users/etienne/Documents/Developer/Iterative_methods/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/etienne/Documents/Developer/Iterative_methods /Users/etienne/Documents/Developer/Iterative_methods /Users/etienne/Documents/Developer/Iterative_methods/build /Users/etienne/Documents/Developer/Iterative_methods/build /Users/etienne/Documents/Developer/Iterative_methods/build/CMakeFiles/2_jacobi_weak_scalability.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/2_jacobi_weak_scalability.dir/depend

