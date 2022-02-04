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

#include "Esn.h"
#include "EsnOpts.h"

#include <algorithm>
#include <armadillo>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <random>

//_____________________________________________________________________________________________________________________

void Esn::BuildNetwork( )
{
    gridSize = opts.leakingRates.size() + opts.spectralRadii.size() + opts.regularizations.size() + opts.inputScalings.size();

    // fill vector with random, sparse numbers and shuffle //
    int numZeroElements = std::round( opts.reservoirSize * opts.reservoirSize  * (opts.sparsity) );
    std::random_device rd;
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<double> dist(-0.5, 0.5);
    arma::vec vtemp( opts.reservoirSize * opts.reservoirSize );
    vtemp.imbue( [&]() { return dist(gen); } );
    vtemp.subvec(0, numZeroElements-1).fill(0.0);
    vtemp = arma::shuffle( vtemp );

    // copy to reservoir //
    weights.res.set_size( opts.reservoirSize, opts.reservoirSize );
    weights.res = arma::reshape(vtemp, opts.reservoirSize, opts.reservoirSize );

    // store largest eigenvalue  //
    arma::cx_vec eigval; arma::cx_mat eigvec;
    arma::eig_gen(eigval, eigvec, weights.res);
    arma::vec eigvalReal = arma::real( eigval );
    eigvalReal = arma::abs( eigvalReal );
    weights.resMaxEigenvalue = eigvalReal.max();

    /// prepare input layer ///
    weights.in.set_size( opts.reservoirSize, 2 );
    weights.in.imbue( [&]() { return dist(gen); } );
} 
//_____________________________________________________________________________________________________________________

Esn::Esn( EsnOpts esnOpts ) 
{
  opts = esnOpts;
}
//_____________________________________________________________________________________________________________________

void Esn::DriveNetwork( const arma::vec& input, bool shedRows )
{
    float leakingRate = GetLeakingRate();

    predicted.set_size( input.size() );
    predicted.fill( 0.0 );
   
    arma::vec vtemp1( 2, arma::fill::ones );
    arma::vec vtemp2( opts.reservoirSize + 2, arma::fill::ones );
    arma::vec x( opts.reservoirSize, arma::fill::ones );

    float u = input(0);
    float vsize = input.size()-1;
    for ( int i = 0; i < vsize; ++i ) {
        vtemp1[1] = u;
        x = (1 - leakingRate)*x + leakingRate * arma::tanh( weights.inScaled * vtemp1 + weights.resScaled * x );
        vtemp2[1] = u;
        vtemp2.subvec(2, weights.x.n_rows-1) = x;
        predicted(i) = arma::conv_to< double >::from( weights.out*vtemp2 );
        u = input( i+1 );
    }

    vsize = vsize + 1;
    if ( shedRows ) {
        RemoveWashoutAndLastKPredictionsByRows( predicted, vsize, trialLength );
    }
} 
//_____________________________________________________________________________________________________________________

float Esn::GetBestInputScaling() { return weightsBest.opts[0]; }
//_____________________________________________________________________________________________________________________

float Esn::GetBestLeakingRate() { return weightsBest.opts[2]; }
//_____________________________________________________________________________________________________________________

float Esn::GetBestRegularization() { return weightsBest.opts[3]; }
//_____________________________________________________________________________________________________________________

float Esn::GetBestSpectralRadius() { return weightsBest.opts[1]; }
//_____________________________________________________________________________________________________________________

float Esn::GetBestValidationError() { return weightsBest.opts[4]; }
//_____________________________________________________________________________________________________________________

float Esn::GetInputScaling() { return weights.opts[0]; }
//_____________________________________________________________________________________________________________________

float Esn::GetLeakingRate() { return weights.opts[2]; }
//_____________________________________________________________________________________________________________________

void Esn::GetOutputWeights()
{
    float regularization = GetRegularization();
    arma::mat identityMatrix( opts.reservoirSize+2, opts.reservoirSize+2, arma::fill::eye );

    //weights.out = dataTrainTarget.t() * weights.x.t() * arma::inv( weights.x * weights.x.t() 
    //               + regularization*identityMatrix );

    arma::mat m1 = weights.x * weights.x.t() + regularization * identityMatrix;
    arma::mat m2 = weights.x * dataTrainTarget;
    arma::mat m3 = arma::solve( m1, m2 );
    weights.out = m3.t();
}
//_____________________________________________________________________________________________________________________

float Esn::GetSpectralRadius() { return weights.opts[1]; }
//_____________________________________________________________________________________________________________________

float Esn::GetRegularization() { return weights.opts[3]; }
//_____________________________________________________________________________________________________________________

void Esn::GetTargetData( arma::vec& data, arma::vec& dataTarget )
{
    int numTimepoints = data.n_elem;
    dataTarget.resize( numTimepoints );
    dataTarget.fill( 0.0 );
    dataTarget.subvec( 0, numTimepoints - opts.steps ) = data.subvec( opts.steps  - 1, numTimepoints - 1 );
}
//_____________________________________________________________________________________________________________________

float Esn::GetValidationError()
{
    if ( dataValTarget.size() == 0 ) {
        return -1;
    }
    float a = arma::sum( arma::pow( ( dataValTarget - predicted ), 2 ) );
    float b = arma::sum( arma::pow( ( dataValTarget - arma::mean(dataValTarget) ), 2 ) );
    return std::sqrt( a/b ) * 100;
}
//_____________________________________________________________________________________________________________________

int Esn::IsBadInputOrRunOptions()
{
    int isBad = 0;

    if ( dataTrain.size() == 0 ) {
        std::cerr << "ERROR: training data not loaded -- " << opts.trainFilename << std::endl;
        isBad = 1;
    }

    if ( !opts.validationFilename.empty() && dataVal.size() == 0 ) {
        std::cerr << "ERROR: validation data not loaded";
        isBad = 1;
    }

    if ( !opts.testFilename.empty() && dataTest.size() == 0 ) {
        std::cerr << "ERROR: test data not loaded";
        isBad = 1;
    }
    
    if ( dataVal.size() == 0 && ( opts.leakingRates.size() > 1 || opts.inputScalings.size() > 1 ||
                                  opts.regularizations.size() > 1 || opts.spectralRadii.size() > 1 ) ) {
        std::cout << "ERROR: validation data is required if using multiple values for a given option" << std::endl;
        isBad = 1;
    }

    return isBad;
}
//_____________________________________________________________________________________________________________________

void Esn::LoadAllData()
{
    std::cout << "Loading data..." << std::flush;
 
    for ( int f = 0; f<3; ++f ) {
        if ( f == 0 ) { 
            int dataTrainSize = LoadData( opts.trainFilename, dataTrain );    
            if ( dataTrainSize > 0 ) {      
                GetTargetData( dataTrain, dataTrainTarget );
                RemoveWashoutAndLastKPredictionsByRows( dataTrainTarget, dataTrainSize, trialLength );
            }
        } 
        else if ( f == 1 ) { 
            int dataValSize = LoadData(  opts.validationFilename, dataVal );
            if  ( dataValSize > 0 ) {
                GetTargetData( dataVal, dataValTarget );
                RemoveWashoutAndLastKPredictionsByRows( dataValTarget, dataValSize, trialLength ); 
            }
        } 
        else if ( f == 2 && !opts.testFilename.empty() ) {
            int dataTestSize = LoadData( opts.testFilename, dataTest );   
        }            
     }
   
    std::cout << " done" << std::endl << std::flush;
} 
//_____________________________________________________________________________________________________________________

int Esn::LoadData( std::string fn, arma::vec& data )
{
    int numTimepoints = 0;

    std::ifstream is;
    is.open( fn, std::ios::binary  | std::ios::in );
    if ( is ) { 
        double val;
        
        is.read( (char*)&val, sizeof(double) ); 
        trialLength = val;

        is.read( (char*)&val, sizeof(double) ); 
        numTimepoints = trialLength * val;

        data.set_size( numTimepoints );
        for ( int i=0; i<numTimepoints; ++i ) {
            is.read( (char*)&val, sizeof(double) );
            data(i) = val;
        } 
    }
    else {
        numTimepoints = 0;
        trialLength = 0;
    }

    return numTimepoints;
}
//_____________________________________________________________________________________________________________________

void Esn::RemoveWashoutAndLastKPredictionsByRows( arma::vec& data, int numTimepoints, int trialLength) {
for ( int i = numTimepoints - trialLength; i >= 0; i = i - trialLength ) {
    data.shed_rows( i+trialLength-opts.steps -1, i+trialLength-1 ); 
    data.shed_rows( i+0, i+opts.washout -1 ); 
    }
}
//_____________________________________________________________________________________________________________________

int Esn::Run()
{
    LoadAllData(); 

     if ( IsBadInputOrRunOptions() ) {
        return 1;
    }

	for ( int i=0; i<opts.numNetworks; ++i) {
		Train();
	}

    WriteParameters();

    if ( !opts.testFilename.empty() ) {
        Test();
        WritePredictions();
    }

    return 0;
}
//_____________________________________________________________________________________________________________________

void Esn::SetInputScaling( float is )
{
    weights.inScaled = weights.in * is;
    weights.opts[0] = is;;
}
//_____________________________________________________________________________________________________________________

void Esn::SetSpectralRadius( float sr )
{
    weights.resScaled = weights.res * sr /  weights.resMaxEigenvalue;
    weights.opts[1] = sr;
}
//_____________________________________________________________________________________________________________________

void Esn::SetLeakingRate( float lr ) { weights.opts[2] = lr; }
//_____________________________________________________________________________________________________________________

void Esn::SetRegularization( float reg ) { weights.opts[3] = reg; }
//_____________________________________________________________________________________________________________________

void Esn::Test()
{
    std::cout << "Generating predictions... "  << std::flush; 
    weights = weightsBest;
    DriveNetwork( dataTest, false ); 
    std::cout << " done" << std::flush << std::endl;
}
//_____________________________________________________________________________________________________________________

void Esn::Train() {
    std::cout << "Training network..." << std::endl << std::flush;
    
    /// randomly generate network weights ///
    BuildNetwork();
    
    /// drive reservoir and collect States ///
    int dataValSize = dataVal.size();
    float is, sr, reg, lr;
    for ( int i=0; i<opts.inputScalings.size(); ++i ) {
        is = opts.inputScalings[i];
        SetInputScaling( is );
        for ( int s=0; s<opts.spectralRadii.size(); ++s ) {
            sr = opts.spectralRadii[s];
            SetSpectralRadius( sr ); 
            for ( int l=0; l<opts.leakingRates.size(); ++l ) {
                lr = opts.leakingRates[l];
                SetLeakingRate( lr ); 
                
                /// prepare state matrix ///
                weights.x.set_size( opts.reservoirSize + 2, dataTrain.size() );
                weights.x.fill( 0.0 );

                arma::vec x( opts.reservoirSize, arma::fill::ones );
                arma::vec vtemp( 2, arma::fill::ones );
                float vsize = dataTrain.size();
                for ( int i = 0; i < vsize; ++i ) {
                    vtemp[1] = dataTrain(i);
                    x = (1 - lr )*x + lr * arma::tanh( weights.inScaled * vtemp + weights.resScaled * x );
                    weights.x(1,i) = vtemp[1];
                    weights.x.col(i).subvec(2, weights.x.n_rows-1) = x;
                } 

                for ( int i = vsize - trialLength; i >= 0; i = i - trialLength ) {
                    weights.x.shed_cols( i+trialLength-opts.steps -1, i+trialLength-1 ); 
                    weights.x.shed_cols( i+0, i+opts.washout -1 ); 
                }
            
                for ( int r=0; r<opts.regularizations.size(); ++r ) {
                    reg = opts.regularizations[r];
                    SetRegularization( reg );
                    GetOutputWeights();
                    if ( dataVal.size() > 0 ) {
                        DriveNetwork( dataVal, true );
                    }

                    if ( dataValSize > 0 ) {
                        float valError = GetValidationError();
                        weights.opts[4] = valError;

                        float valErrorBest =  weightsBest.opts[4];
                        if ( valErrorBest == -1 || valError < valErrorBest )
                        {
                            weightsBest = weights;
                        }
                        std::cout << "  " <<  valError << " : " << lr << "," << sr << "," 
                                          << is << "," << reg << std::endl << std::flush;
                    }
                    else {
                        weightsBest = weights;
                    }
                }
            }
        }
    }
    std::cout << "  done" << std::flush << std::endl;
}
//_____________________________________________________________________________________________________________________

void Esn::WriteParameters()
{
    std::string fn = opts.outputDirectory +  "/esn_parameters.txt";
    std::ofstream os;

    os.open( fn );
    os << GetBestLeakingRate() << ", ";
    os << GetBestSpectralRadius() << ", ";
    os << GetBestInputScaling() << ", ";
    os << GetBestRegularization() << ", ";
    os.close();
}
//_____________________________________________________________________________________________________________________

int Esn::WritePredictions()
{
    std::cout << "Writing predictions..." << std::flush;

    std::string fn = opts.outputDirectory + "/esn_prediction.bin";
    std::ofstream os;
    double d;

    os.open( fn, std::ios::binary  | std::ios::out );
    if ( !os ) { std::cerr << "ERROR: opening " << fn << std::endl; return 1; }
    d = trialLength;
    os.write( (char*)&d, sizeof(double)  );
    d = predicted.size() / trialLength;
    os.write( (char*)&d, sizeof(double)  );
    d = opts.steps;
    os.write( (char*)&d, sizeof(double)  );
    
    int vsize = predicted.size();
    for ( int i = 0; i < vsize; ++i ) {
        d = predicted[i]; os.write( (char*)&d, sizeof(double)  );
    }
    os.close( );

   
    std::cout << " done" << std::flush << std::endl;
    return 0;
}
//_____________________________________________________________________________________________________________________