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
#include <iostream>

#include "Esn.h"
#include "EsnOpts.h"

//_____________________________________________________________________________________________________________________

void PrintUsage( const char* argv[] )
{
	std::cerr << std::endl;
   std::cerr << "Copyright (C) 2022 Erin Gibson"  << std::endl;
	std::cerr << "This program comes with ABSOLUTELY NO WARRANTY" << std::endl;
	std::cerr << "This is free software, and you are welcome to redistribute it" << std::endl;
	std::cerr << "under certain conditions; see license file for details." << std::endl;
	std::cerr << std::endl;
	std::cerr << "Usage: " << std::filesystem::path( argv[0] ).filename().string() << " <Options>" << std::endl;
	std::cerr << "Options:" << std::endl;
	std::cerr << "  -t : train data filename" << std::endl;
	std::cerr << "  -v : validation data filename" << std::endl;
	std::cerr << "  -p : test data filename" << std::endl;
	std::cerr << "  -d : output directory" << std::endl;
	std::cerr << "  -l : leaking rate" << std::endl;
	std::cerr << "  -r : regularization" << std::endl;
	std::cerr << "  -s : spectral radius" << std::endl;
	std::cerr << "  -i : input scaling" << std::endl;
	std::cerr << "  -k : number of steps" << std::endl;
	std::cerr << "  -w : washout" << std::endl;
	std::cerr << "  -n : reservoir size" << std::endl;
	std::cerr << "  -c : connection sparsity" << std::endl;
	std::cerr << "  -x : number of random initializations" << std::endl;
	std::cerr << "Notes:" << std::endl;
	std::cerr << "  -the -l -r -s -i options can be specified more than once," << std::endl;
	std::cerr << "   and validation data will be used to find the optimal value" << std::endl;
	std::cerr << "   i.e. -l 0.2 -l 0.4 -l 0.6 -l 0.8 etc." << std::endl;
	std::cerr << "  -can use -a and -b to find optimal parameters, and then" << std::endl;
	std::cerr << "   use -a and -c to predict" << std::endl;
	std::cerr << std::endl;	
}
//_____________________________________________________________________________________________________________________

int main ( const int argc, const char* argv[] ) 
{
	if ( argc == 1 ) {
		PrintUsage( argv );
		exit( 1 );
	}

	EsnOpts esnOpts;
   if ( esnOpts.GetInputOpts( argc, argv ) ) {
		exit( 1 );
	}

	Esn esn( esnOpts );
	esn.Run();

	return 0;
}
//_____________________________________________________________________________________________________________________