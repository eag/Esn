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

#ifndef ESN_H_
#define ESN_H_

#include <string>
#include <vector>
#include <memory>
#include <fstream>

#include <armadillo>

#include "EsnOpts.h"
#include "EsnWeights.h"

//_____________________________________________________________________________________________________________________

class Esn
{
    private:
        arma::vec actual;
        arma::vec dataTrain;
        arma::vec dataTrainTarget;
        arma::vec dataTest;
        arma::vec dataVal;
        arma::vec dataValTarget;
        arma::vec predicted;
        arma::vec yt;

        EsnOpts opts;

        EsnWeights weights;
        EsnWeights weightsBest;
 
        int gridSize;
        int trialLength;

        void  BuildNetwork();
        void  DriveNetwork( const arma::vec&, bool );
        float GetBestInputScaling();
        float GetBestLeakingRate();
        float GetBestValidationError();
        float GetBestRegularization();
        float GetBestSpectralRadius();
        float GetInputScaling();
        float GetLeakingRate();
        void  GetOutputWeights();
        void  GetTargetData( arma::vec&, arma::vec&  );
        float GetRegularization();
        float GetSpectralRadius();
        float GetValidationError();
        void  LoadAllData();
        int   LoadData( std::string, arma::vec& );
        int   IsBadInputOrRunOptions();
        void  RemoveWashoutAndLastKPredictionsByRows( arma::vec&, int, int);
        void  SetInputScaling( float );
        void  SetLeakingRate( float );
        void  SetRegularization( float );
        void  SetSpectralRadius( float );
        void  Train();
        void  Test();
        void  WriteParameters();
        int   WritePredictions();

    public:

        Esn( EsnOpts );

        int Run();

};

#endif
//_____________________________________________________________________________________________________________________