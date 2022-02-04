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


#ifndef ESNOPTS_H_
#define ESNOPTS_H_

#include <vector>
#include <string>

//_____________________________________________________________________________________________________________________

class EsnOpts
{
	public:

		std::string trainFilename = "";
		std::string testFilename = "";
		std::string validationFilename = "";
		std::string outputDirectory = "";

		std::vector< float > leakingRates;
		std::vector< float > regularizations;
		std::vector< float > inputScalings;
		std::vector< float > spectralRadii;
		
		int   steps             = -1;
		int   washout           = -1;
		float sparsity          = 0.85;
		int   reservoirSize     = 200;
		int   numNetworks       = 3;

		int GetInputOpts( const int, const char*[] );

	private:

		int  CheckAndPrintNumericOpts( const float , const std::string&  );
		int  CheckAndPrintVectorOpts( const std::vector< float >&, const std::string& );
		int  CheckFilenameOpts();
		void PrintFilenameOpts( const std::string&, const std::string& );
};

#endif
//_____________________________________________________________________________________________________________________
