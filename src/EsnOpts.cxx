/*
Copyright (C) 2022 Erin Gibson

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
//_____________________________________________________________________________________________________________________


#include <filesystem>
#include "EsnOpts.h"
#include "cxxopts.hpp"

//_____________________________________________________________________________________________________________________

int EsnOpts::GetInputOpts( const int numInputOpts, const char* inputOpts[] )
{
	try {
		cxxopts::Options options( inputOpts[0] );
		options.add_options()
		( "t", "train filename",      cxxopts::value( trainFilename ) )
		( "p", "test filename",       cxxopts::value( testFilename ) )
		( "v", "validation filename", cxxopts::value( validationFilename ) )
		( "d", "output directory",    cxxopts::value( outputDirectory ) )
		( "l", "leaking rate",        cxxopts::value( leakingRates ) )
		( "r", "regularization",      cxxopts::value( regularizations ) )
		( "s", "spectral radius",     cxxopts::value( spectralRadii ) )
		( "i", "input scaling",       cxxopts::value( inputScalings ) )
		( "k", "number of steps",     cxxopts::value( steps ) )
		( "w", "washout",             cxxopts::value( washout ) )
		( "n", "reservoir size",      cxxopts::value( reservoirSize ) )
		( "c", "connection sparsity", cxxopts::value( sparsity ) )
		( "x", "number of random initializations", cxxopts::value( numNetworks ) )
		;
		options.parse(numInputOpts, inputOpts);
	}
	catch(const cxxopts::OptionException& e) {
		std::cerr << "ERROR: parsing option " << e.what() << std::endl; return 1;
	}

	std::cout << "Network parameters:" << std::endl;
	PrintFilenameOpts( trainFilename, "train filename");
	PrintFilenameOpts( testFilename, "test filename");
	PrintFilenameOpts( validationFilename, "validation filename");
	PrintFilenameOpts( outputDirectory, "output directory");
	if ( CheckAndPrintVectorOpts( inputScalings,   "input scaling"   ) ) { return 1; }
	if ( CheckAndPrintVectorOpts( spectralRadii,   "spectral radii"  ) ) { return 1; }
	if ( CheckAndPrintVectorOpts( leakingRates,    "leaking rates"   ) ) { return 1; }
	if ( CheckAndPrintVectorOpts( regularizations, "regularizations" ) ) { return 1; }
	if ( CheckAndPrintNumericOpts( steps,   "steps"   ) ) { return 1; }
	if ( CheckAndPrintNumericOpts( washout, "washout" ) ) { return 1; }
	if ( CheckAndPrintNumericOpts( sparsity, "sparsity" ) ) { return 1; }
	if ( CheckAndPrintNumericOpts( reservoirSize, "reservoir size" ) ) { return 1; }
	if ( CheckAndPrintNumericOpts( numNetworks, "number of random initializations" ) ) { return 1; }
	if ( CheckFilenameOpts() ) { return 1; }
	
	return 0;
}

int EsnOpts::CheckAndPrintNumericOpts( const float input, const std::string& msg )
{
	if ( input <= 0) {
		std::cerr << "ERROR: no " << msg << " specified" << std::endl;
		return 1;
	}
	
	std::cout << "  " << msg << " -- " << input << std::endl;;
   return 0;
}

int EsnOpts::CheckAndPrintVectorOpts( const std::vector< float >& input, const std::string& msg )
{
	int vsize = input.size();
	if ( vsize <=0 ) {
		std::cerr << "ERROR: no " << msg << " specified" << std::endl;
		return 1;
	}

	std::cout << "  " << msg << " -- ";
	for ( int i = 0; i < vsize; ++i ) {
		std::cout << input[i] << " ";
	}
	std::cout << std::endl;
	return 0;
}

int EsnOpts::CheckFilenameOpts()
{
	if ( trainFilename.empty() ) {
		std::cerr << "ERROR: must supply -t option" << std::endl;
		return 1;
	}

	if ( validationFilename.empty() && testFilename.empty() ) {
		std::cerr << "ERROR: must supply -v and/or -p options" << std::endl;
		return 1;
	}

	if ( outputDirectory.empty() ) {
		std::cerr << "ERROR: must supply -d option" << std::endl;
		return 1;
	}

	return 0;
}

void EsnOpts::PrintFilenameOpts( const std::string& filename, const std::string& msg )
{
	std::cout << "  "  << msg << " -- " 
	          << std::filesystem::path( filename ).filename().string() << std::endl;
}

//_____________________________________________________________________________________________________________________