# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/fmw-sa/Documents/cv_trial

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/fmw-sa/Documents/cv_trial

# Include any dependencies generated for this target.
include CMakeFiles/down_sampling.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/down_sampling.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/down_sampling.dir/flags.make

CMakeFiles/down_sampling.dir/down_sampling.cpp.o: CMakeFiles/down_sampling.dir/flags.make
CMakeFiles/down_sampling.dir/down_sampling.cpp.o: down_sampling.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/fmw-sa/Documents/cv_trial/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/down_sampling.dir/down_sampling.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/down_sampling.dir/down_sampling.cpp.o -c /home/fmw-sa/Documents/cv_trial/down_sampling.cpp

CMakeFiles/down_sampling.dir/down_sampling.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/down_sampling.dir/down_sampling.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/fmw-sa/Documents/cv_trial/down_sampling.cpp > CMakeFiles/down_sampling.dir/down_sampling.cpp.i

CMakeFiles/down_sampling.dir/down_sampling.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/down_sampling.dir/down_sampling.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/fmw-sa/Documents/cv_trial/down_sampling.cpp -o CMakeFiles/down_sampling.dir/down_sampling.cpp.s

CMakeFiles/down_sampling.dir/down_sampling.cpp.o.requires:
.PHONY : CMakeFiles/down_sampling.dir/down_sampling.cpp.o.requires

CMakeFiles/down_sampling.dir/down_sampling.cpp.o.provides: CMakeFiles/down_sampling.dir/down_sampling.cpp.o.requires
	$(MAKE) -f CMakeFiles/down_sampling.dir/build.make CMakeFiles/down_sampling.dir/down_sampling.cpp.o.provides.build
.PHONY : CMakeFiles/down_sampling.dir/down_sampling.cpp.o.provides

CMakeFiles/down_sampling.dir/down_sampling.cpp.o.provides.build: CMakeFiles/down_sampling.dir/down_sampling.cpp.o

# Object files for target down_sampling
down_sampling_OBJECTS = \
"CMakeFiles/down_sampling.dir/down_sampling.cpp.o"

# External object files for target down_sampling
down_sampling_EXTERNAL_OBJECTS =

down_sampling: CMakeFiles/down_sampling.dir/down_sampling.cpp.o
down_sampling: CMakeFiles/down_sampling.dir/build.make
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
down_sampling: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
down_sampling: CMakeFiles/down_sampling.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable down_sampling"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/down_sampling.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/down_sampling.dir/build: down_sampling
.PHONY : CMakeFiles/down_sampling.dir/build

CMakeFiles/down_sampling.dir/requires: CMakeFiles/down_sampling.dir/down_sampling.cpp.o.requires
.PHONY : CMakeFiles/down_sampling.dir/requires

CMakeFiles/down_sampling.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/down_sampling.dir/cmake_clean.cmake
.PHONY : CMakeFiles/down_sampling.dir/clean

CMakeFiles/down_sampling.dir/depend:
	cd /home/fmw-sa/Documents/cv_trial && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/fmw-sa/Documents/cv_trial /home/fmw-sa/Documents/cv_trial /home/fmw-sa/Documents/cv_trial /home/fmw-sa/Documents/cv_trial /home/fmw-sa/Documents/cv_trial/CMakeFiles/down_sampling.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/down_sampling.dir/depend

