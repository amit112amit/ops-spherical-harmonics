#include <ctime>
#include <string>
#include <algorithm>
#include <iostream>
#include <random>
#include <math.h>
#include <vtkNew.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkCellLocator.h>
#include <vtkDelaunay3D.h>
#include <vtkDataSetSurfaceFilter.h>
#include <vtkPolyData.h>
#include <vtkDoubleArray.h>
#include <vtkPoints.h>
#include <vtkIdFilter.h>
#include <vtkPointData.h>
#include <vtkCellArray.h>
#include <vtkIdList.h>
#include <vtkLinearSubdivisionFilter.h>
#include <vtkGenericCell.h>
#include <vtkVertexGlyphFilter.h>
#include <vtkDataObject.h>
#include <Eigen/Dense>

extern "C" void shexpandlsq_wrapper_(double* cilm, double* d, double* lat,
	double* lon, int* N, int* lmax, double* chi2);

extern "C" void makegridpoints_wrapper_(double* cilm, int* lmax, int* n,
	double* lat, double* lon, double* points, int* dealloc);

extern "C" void glqgridcoord_wrapper_(double* latglq, double* longlq,
	int* lmax, int* nlat, int* nlong);

extern "C" void shglq_wrapper_(int* lmax, double* zero, double* w,
	double* plx);

extern "C" void shexpandglq_wrapper_(double* cilm, int* lmax, double* gridglq,
	double* w, double* plx);

extern "C" void makegridglq_wrapper_(double* gridglq, double* cilm, int* lmax,
	double* plx);

extern "C" void shexpanddh_wrapper_(double* grid, int* n, double* cilm,
	int* lmax);

extern "C" void makegriddh_wrapper_(double* grid, int* n, double* cilm,
	int* lmax);

extern "C" void shpowerspectrum_wrapper_(double* cilm, int* lmax,
	double* pspectrum);

extern "C" void plmon_wrapper_(double* p, int* lmax, double* z);

int main(){

    // Read the polydata
    vtkNew<vtkPolyDataReader> reader;
    reader->SetFileName("T7.vtk");
    reader->Update();
    auto pointCloud = reader->GetOutput();
    int N = pointCloud->GetNumberOfPoints();

    // Copy the point coordinates into a matrix
    Eigen::Map<Eigen::Matrix3Xd> ptsMat((double_t*)pointCloud->GetPoints()
	    ->GetData()->GetVoidPointer(0),3,N);

    // Test function: 2*Y_1,-1 + Y_10 + 3*Y_11, so l=1, m=-1,0,1
    int lmax = 7;
    Eigen::VectorXd plms( (lmax + 1)*(lmax + 2)/2 );
    auto Plm = [&plms](const int l, const int m){
	return plms( l*(l+1)/2 + m );
    };

    // Calculate points on a sphere of average radius and displacements
    vtkNew<vtkPoints> spherePts;
    vtkNew<vtkDoubleArray> displacements;
    displacements->SetName("Displacements");
    displacements->SetNumberOfComponents(1);
    for(auto i = 0; i < N; ++i){
	Eigen::Vector3d p, q;
	pointCloud->GetPoint(i, &p(0));
	q =  p.normalized();
	spherePts->InsertNextPoint( &q(0) );
	double_t x = q(0);
	double_t y = q(1);
	double_t z = q(2);
	plmon_wrapper_(plms.data(), &lmax, &z);
	auto phi = std::atan2(y,x);
	auto disp = 2*Plm(1,-1)*std::sin(phi) + Plm(1,0) +
	    3*Plm(1,1)*std::cos(phi);
	displacements->InsertNextTuple1(disp);
    }

    // Create a mesh on the sphere
    vtkNew<vtkPolyData> sphere;
    sphere->SetPoints( spherePts );
    sphere->GetPointData()->AddArray( displacements );
    vtkNew<vtkIdFilter> idf;
    idf->SetIdsArrayName( "OrigIds" );
    idf->PointIdsOn();
    idf->SetInputData( sphere );
    vtkNew<vtkDelaunay3D> d3D;
    d3D->SetInputConnection( idf->GetOutputPort() );
    vtkNew<vtkDataSetSurfaceFilter> dssf;
    dssf->SetInputConnection(d3D->GetOutputPort());
    dssf->Update();
    auto final = dssf->GetOutput()->GetPolys();
    auto idArray = dssf->GetOutput()->GetPointData()->GetArray( "OrigIds" );
    final->InitTraversal();
    vtkNew<vtkCellArray> triangles;
    vtkNew<vtkIdList> ids;
    while( final->GetNextCell( ids ) ){
	triangles->InsertNextCell(3);
	for( auto i = 0; i < 3; ++i )
	    triangles->InsertCellPoint( (vtkIdType)idArray->GetTuple1(
			ids->GetId(i) ) );
    }
    sphere->SetPolys( triangles );
    vtkNew<vtkPolyDataWriter> writer;
    writer->SetFileName("TestShape.vtk");
    writer->SetInputData(sphere);
    writer->Write();

    // Create a locator to identify the cells that contain the Gauss quadrature
    // points
    vtkNew<vtkCellLocator> cellLoc;
    cellLoc->SetDataSet( sphere );
    cellLoc->BuildLocator();

    //**********************************************************************//
    // Create the Gauss-Lengendre grid
    int nlat, nlong;
    Eigen::VectorXd latglq(lmax + 1);
    Eigen::VectorXd longlq(2*lmax + 1);
    glqgridcoord_wrapper_(latglq.data(), longlq.data(), &lmax, &nlat, &nlong);

    // Find the cell to which each point of the grid belongs and set its value
    // by linear interpolation
    auto coords = []( const double_t T, const double_t P ){
	Eigen::Vector3d X;
	auto SinT = std::sin(T*M_PI/180.0);
	X << SinT*std::cos(P*M_PI/180.0), SinT*std::sin(P*M_PI/180.0),
	  std::cos( T*M_PI/180.0);
	return X;
    };
    vtkNew<vtkPoints> glqPoints;
    vtkNew<vtkCellArray> glqVerts;
    vtkNew<vtkDoubleArray> glqDisp;
    glqDisp->SetName("GLQDisp");
    glqDisp->SetNumberOfComponents(3);
    Eigen::VectorXd gridglq((lmax + 1)*(2*lmax + 1));
    Eigen::VectorXd plx( (lmax + 1)*(lmax + 1)*(lmax + 2)/2 );
    Eigen::VectorXd w( lmax + 1 ), zero( lmax + 1 );
    Eigen::VectorXd cilm1( 2*(lmax + 1)*(lmax + 1) );
    Eigen::VectorXd pspectrum1( lmax + 1 );
    shglq_wrapper_(&lmax, zero.data(), w.data(), plx.data() );

    clock_t t = clock();
    for(auto z = 0; z < 1; ++z){
	for( auto j = 0; j < 2*lmax + 1; ++j ){
	    for( auto i = 0; i < lmax + 1; ++i ){
		// Get the spherical coordinate
		auto X = coords( 90.0 - latglq(i), longlq(j) );
		// Locate cell
		Eigen::Vector3d closestPoint, pcoords, weights;
		vtkIdType cellId;
		int subId;
		double_t dist2;
		vtkNew<vtkGenericCell> genCell;
		cellLoc->FindClosestPoint( &X(0), &closestPoint(0), genCell,
			cellId, subId, dist2);
		// Get the parametric coordinates and weights
		genCell->EvaluatePosition( &X(0), &closestPoint(0), subId,
			&pcoords(0), dist2, &weights(0) );
		vtkNew<vtkIdList> pointIds;
		sphere->GetCellPoints( cellId, pointIds );
		// Interpolate to find the displacement at quadrature point
		gridglq(i + (lmax + 1)*j) =
		    weights[0]*displacements->GetTuple1(pointIds->GetId(0)) +
		    weights[1]*displacements->GetTuple1(pointIds->GetId(1)) +
		    weights[2]*displacements->GetTuple1(pointIds->GetId(2));
		glqPoints->InsertNextPoint( &X(0) );
		Eigen::Vector3d disp;
		disp = gridglq(i + (lmax + 1)*j)*X.normalized();
		glqDisp->InsertNextTuple( &disp(0) );
	    }
	}
	vtkNew<vtkPolyData> glqBody;
	glqBody->SetPoints( glqPoints );
	glqBody->GetPointData()->SetVectors( glqDisp );
	writer->SetInputData( glqBody );
	writer->SetFileName( "GLQBody.vtk" );
	writer->Write();

	// Expand using Gauss-Legendre Quadrature and get the power spectrum
	shexpandglq_wrapper_( cilm1.data(), &lmax, gridglq.data(), w.data(),
		plx.data());
    }
    t = clock() - t;
    std::cout<< "Time for 1 steps GLQ = " << ((float)t)/CLOCKS_PER_SEC
	<< std::endl;

    shpowerspectrum_wrapper_( cilm1.data(), &lmax, pspectrum1.data() );
    // Cross-check
    Eigen::VectorXd gridglq_out( (lmax + 1)*(2*lmax + 1) );
    makegridglq_wrapper_(gridglq_out.data(), cilm1.data(), &lmax, plx.data());
    std::cout<< "Error norm of GLQ = " << (gridglq - gridglq_out).norm()
	<< std::endl;
    //**********************************************************************//
    // Expand using Driscol-Healy and get the power spectrum
    int ndh = 2*(lmax + 1);
    vtkNew<vtkPoints> DHPoints;
    vtkNew<vtkCellArray> DHVerts;
    vtkNew<vtkDoubleArray> DHDisp;
    DHDisp->SetName("DHGridDisp");
    DHDisp->SetNumberOfComponents(3);
    Eigen::VectorXd gridDH( ndh*ndh );
    Eigen::VectorXd cilm2( ndh*ndh/2 ), pspectrum2(lmax + 1);

    t = clock();
    for(auto z = 0; z < 1; ++z){
	for( auto i = 0; i < ndh; ++i ){
	    auto latDH = i*(180.0/ndh);
	    for( auto j = 0; j < ndh; ++j ){
		// Get the spherical coordinate
		auto lonDH = j*360.0/ndh;
		auto X = coords( latDH, lonDH );
		// Locate cell
		Eigen::Vector3d closestPoint, pcoords, weights, dummy;
		vtkIdType cellId;
		int subId;
		double_t dist2;
		vtkNew<vtkGenericCell> genCell;
		cellLoc->FindClosestPoint( &X(0), &closestPoint(0), genCell,
			cellId, subId, dist2);
		// Get the parametric coordinates and weights
		auto out = genCell->EvaluatePosition( &closestPoint(0), &dummy(0),
			subId, &pcoords(0), dist2, &weights(0) );
		vtkNew<vtkIdList> pointIds;
		sphere->GetCellPoints( cellId, pointIds );
		// Interpolate to find the displacement at quadrature point
		gridDH(i + ndh*j) =
		    weights[0]*displacements->GetTuple1(pointIds->GetId(0)) +
		    weights[1]*displacements->GetTuple1(pointIds->GetId(1)) +
		    weights[2]*displacements->GetTuple1(pointIds->GetId(2));
		DHPoints->InsertNextPoint( &X(0) );
		Eigen::Vector3d disp;
		disp = gridDH(j + ndh*i)*X.normalized();
		DHDisp->InsertNextTuple( &disp(0) );
	    }
	}
	vtkNew<vtkPolyData> DHBody;
	DHBody->SetPoints( DHPoints );
	DHBody->GetPointData()->SetVectors( DHDisp );
	writer->SetInputData( DHBody );
	writer->SetFileName( "DHBody.vtk" );
	writer->Write();

	// Finally expand using DH scheme
	shexpanddh_wrapper_(gridDH.data(), &ndh, cilm2.data(), &lmax);
    }
    t = clock() - t;
    std::cout<< "Time for 1 steps DH = " << ((float)t)/CLOCKS_PER_SEC
	<< std::endl;

    shpowerspectrum_wrapper_(cilm2.data(), &lmax, pspectrum2.data());
    // Cross-check
    Eigen::VectorXd gridDH_out( ndh*ndh );
    makegriddh_wrapper_(gridDH_out.data(), &ndh, cilm2.data(), &lmax);
    std::cout<< "Error norm of DH = " << (gridDH - gridDH_out).norm()
	<< std::endl;
    //**********************************************************************//
    //Now we will use LSQ
    Eigen::VectorXd latLSQ(N), lonLSQ(N), gridLSQ(N);
    Eigen::VectorXd cilm3(2*(lmax+1)*(lmax+1)), pspectrum3(lmax + 1);
    double_t chi2;
    t = clock();
    for(auto z = 0; z < 1; ++z){
	for( auto i = 0; i < N; ++i ){
	    double_t X[3], x, y, z;
	    spherePts->GetPoint(i, X);
	    x = X[0]; y = X[1]; z = X[2];
	    // Calculate latitude and longitude for the points
	    latLSQ(i) = std::atan2(z, sqrt(x*x + y*y)) * 180.0/M_PI;
	    lonLSQ(i) = std::atan2(y, x) * 180.0/M_PI;
	    gridLSQ(i) = displacements->GetTuple1(i);
	}
	shexpandlsq_wrapper_(cilm3.data(), gridLSQ.data(), latLSQ.data(),
		lonLSQ.data(), &N, &lmax, &chi2);
    }
    t = clock() - t;
    std::cout<< "Time for 1 steps LSQ = " << ((float)t)/CLOCKS_PER_SEC
	<< std::endl;

    shpowerspectrum_wrapper_(cilm3.data(), &lmax, pspectrum3.data());
    // Cross-check
    Eigen::VectorXd gridLSQ_out(N);
    int dealloc = 1;
    makegridpoints_wrapper_(cilm3.data(), &lmax, &N, latLSQ.data(),
	    lonLSQ.data(), gridLSQ_out.data(), &dealloc);
    std::cout<< "Error norm of LSQ = " << (gridLSQ_out - gridLSQ).norm()
	<< std::endl;

    //**********************************************************************//
    // Now let's print the power-spectrums
    std::cout<< " Power spectrum from GLQ = " << std::endl
	<< pspectrum1.transpose() << std::endl;
    std::cout<< " Power spectrum from DH = " << std::endl
	<< pspectrum2.transpose() << std::endl;
    std::cout<< " Power spectrum from LSQ = " << std::endl
	<< pspectrum3.transpose() << std::endl;

    // Print the coefficients
    std::cout<< "l\tm\tAlm_GLQ\tAlm_DH\tAlm_LSQ" << std::endl;
    for( int l = 0; l <= lmax; ++l ){
	for( int m = -l; m <=l; ++m ){
	    int i = m < 0? 1 : 0;
	    int n = m < 0? -m: m;
	    std::cout<< l << "\t" << m << "\t" 
		<< cilm1( i + 2*(l + n*(lmax + 1)) )<< "\t"
		<< cilm2( i + 2*(l + n*(lmax + 1)) )<< "\t"
		<< cilm3( i + 2*(l + n*(lmax + 1)) )<< "\t"
		<< std::endl;
	}
    }

    //**********************************************************************//
    // Now we will subdivide the sphere polydata and calculate the
    // displacements at more points using the cilm calculated above. We will
    // print all three vtk surfaces and see which is the best.
    vtkNew<vtkLinearSubdivisionFilter> lsdf;
    lsdf->SetInputData( sphere );
    lsdf->SetNumberOfSubdivisions(2);
    lsdf->Update();
    auto sphere2 = lsdf->GetOutput();
    int N2 = sphere2->GetNumberOfPoints();
    Eigen::VectorXd lat2(N2), lon2(N2), grid2(N2);
    for( auto i = 0; i < N2; ++i ){
	double_t X[3], x, y, z;
	sphere2->GetPoint(i, X);
	x = X[0]; y = X[1]; z = X[2];
	// Calculate latitude and longitude for the points
	lat2(i) = std::atan2(z, sqrt(x*x + y*y)) * 180.0/M_PI;
	lon2(i) = std::atan2(y, x) * 180.0/M_PI;
    }
    auto writeResult = [&](Eigen::VectorXd &C, std::string file){
	// First get the deformation vlaues
	makegridpoints_wrapper_(C.data(), &lmax, &N2, lat2.data(),
		lon2.data(), grid2.data(), &dealloc);
	// Add the deformations to average radius along each point's direction
	vtkNew<vtkPoints> newPoints;
	for( auto i = 0; i < N2; ++i ){
	    Eigen::Vector3d X, Xpd;
	    sphere2->GetPoint( i, &X(0) );
	    Xpd = (1.0 + grid2(i))*X.normalized();
	    newPoints->InsertNextPoint( &Xpd(0) );
	}
	vtkNew<vtkPolyData> temp;
	temp->SetPoints( newPoints );
	vtkNew<vtkVertexGlyphFilter> vgf;
	vgf->SetInputData( temp );
	writer->SetInputConnection( vgf->GetOutputPort() );
	writer->SetFileName( file.c_str() );
	writer->Update();
	writer->Write();
    };

    //writeResult( cilm1, "LSQ_Result.vtk" );
    //writeResult( cilm2, "GLQ_Result.vtk" );
    //writeResult( cilm3, "DH_Result.vtk" );

    return 0;
}
