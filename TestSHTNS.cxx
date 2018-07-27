#include <ctime>
#include <fftw3.h>
#include <string>
#include <algorithm>
#include <iostream>
#include <random>
#include <math.h>
#include <vtkNew.h>
#include <vtkPolyDataReader.h>
#include <vtkPolyDataWriter.h>
#include <vtkPolyData.h>
#include <vtkDoubleArray.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkIdList.h>
#include <Eigen/Dense>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include "shtns.h"

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_2<unsigned,K> Vb;
typedef CGAL::Triangulation_data_structure_2<Vb> Tds;
typedef CGAL::Delaunay_triangulation_2<K,Tds> Delaunay;
typedef Delaunay::Face_circulator Face_circulator;
typedef Delaunay::Face_handle Face_handle;
typedef Delaunay::Point Point;
typedef Eigen::Vector3d Vector3d;
typedef Eigen::VectorXd VectorXd;
typedef Eigen::Matrix3d Matrix3d;
typedef Eigen::Matrix3Xd Matrix3Xd;
typedef Eigen::Map<Matrix3Xd> Map3Xd;

int main(){

    // Read the polydata
    vtkNew<vtkPolyDataReader> reader;
    reader->SetFileName("T7.vtk");
    reader->Update();
    auto pointCloud = reader->GetOutput();
    int N = pointCloud->GetNumberOfPoints();

    //**********************************************************************//
    //SHTns: Generate quadrature points
    int lmax = std::floor( sqrt(N) - 1 );
    int mmax = lmax, nlat = 0, nphi = 0, mres = 1;
    shtns_cfg shtns;
    shtns_verbose(1);
    shtns_norm norm = static_cast<shtns_norm>(SHT_REAL_NORM | SHT_NO_CS_PHASE);
    shtns = shtns_create(lmax, mmax, mres, norm );
    shtns_type flags = static_cast<shtns_type>( sht_gauss | SHT_NATIVE_LAYOUT);
    shtns_set_grid_auto(shtns, flags, 1e-14, 1, &nlat, &nphi );
    int NLM = shtns->nlm;
    int numQ = nphi*nlat;

    // Memory allocation
    double_t *f;
    f = static_cast<double_t*>(
	    fftw_malloc( NSPAT_ALLOC(shtns)*sizeof(double_t) ));
    std::complex<double_t> *Slm;
    Slm = static_cast<std::complex<double_t>*>(
	    fftw_malloc( NLM*sizeof(std::complex<double_t>) ));
    // Memory initialization
    for( auto lm = 0; lm < NLM; ++lm){
	Slm[lm] = 0.0;
    }

    // Calculate the quad point Q0
    Eigen::Matrix3Xd Q0(3,numQ);
    for( auto ip = 0; ip < nphi; ++ip){
	for( auto it = 0; it < nlat; ++it){
	    auto phi = ip*M_PI*2/nphi;
	    Q0.col(it + ip*nlat) << shtns->st[it]*std::cos(phi),
		shtns->st[it]*std::sin(phi), shtns->ct[it];
	}
    }

    //************************* Some handy functions ***********************//

    // A lambda function to write a Eigen::Matrix3Xd to VTK file
    auto writeMatToVTK =
	[](Eigen::Matrix3Xd M, std::string file, bool Proj = false){
	    vtkNew<vtkPoints> pts;
	    vtkNew<vtkCellArray> verts;
	    for( auto b = 0; b < M.cols(); ++b ){
		double x, y, z;
		x = M(0,b);
		y = M(1,b);
		z = Proj? 0.0 : M(2,b);
		pts->InsertNextPoint(x, y, z);
		verts->InsertNextCell( 1 );
		verts->InsertCellPoint( b );
	    }
	    vtkNew<vtkPolyData> poly;
	    poly->SetPoints( pts );
	    poly->SetVerts( verts );
	    vtkNew<vtkPolyDataWriter> wr;
	    wr->SetInputData( poly );
	    wr->SetFileName( file.c_str() );
	    wr->Write();
	};

    //**********************************************************************//
    clock_t t = clock();

    for( auto z = 0; z < 1000; ++z ){
	// Copy the point coordinates into a matrix
	Eigen::Map<Eigen::Matrix3Xd> points((double_t*)pointCloud->GetPoints()
		->GetData()->GetVoidPointer(0),3,N);

	// Project points to unit sphere
	points.colwise().normalize();

	// Reset the center of the sphere to origin by translating
	Vector3d center = points.rowwise().mean();
	points = points.colwise() - center;

	// Write original points to VTK
	//writeMatToVTK( points, "BasePoints.vtk" );

	// Rotate all points so that the point in 0th column is along z-axis
	Vector3d c = points.col(0);
	double_t cos_t = c(2);
	double_t sin_t = std::sqrt( 1 - cos_t*cos_t );
	Vector3d axis;
	axis << c(1), -c(0), 0.;
	Matrix3d rotMat, axis_cross, outer;
	axis_cross << 0. , -axis(2), axis(1),
		   axis(2), 0., -axis(0),
		   -axis(1), axis(0), 0.;

	outer.noalias() = axis*axis.transpose();

	rotMat = cos_t*Matrix3d::Identity() + sin_t*axis_cross + (1-cos_t)*outer;
	Matrix3Xd rPts(3,N);
	rPts = rotMat*points; // The points on a sphere rotated

	// Write the rotated points to VTK
	//writeMatToVTK( rPts, "RotBasePoints.vtk" );

	// Calculate the stereographic projections
	Vector3d p0;
	Map3Xd l0( &(rPts(0,1)), 3, N-1 );
	Matrix3Xd l(3,N-1), proj(3,N-1);
	p0 << 0,0,-1; // Point on the plane of projection
	c = rPts.col(0); // The point from which we are projecting
	l = (l0.colwise() - c).colwise().normalized(); // directions of projections
	for( auto j=0; j < N-1; ++j ){
	    proj.col(j) = ((p0(2) - l0(2,j))/l(2,j))*l.col(j) + l0.col(j);
	}

	// Write the rotated points to VTK
	//writeMatToVTK( proj, "ProjBasePoints.vtk", true );

	// Insert the projected points in a CGAL vertex_with_info vector
	std::vector< std::pair< Point, unsigned> > verts;
	for( auto j=0; j < N-1; ++j ){
	    verts.push_back(std::make_pair(Point(proj(0,j),proj(1,j)),j+1));
	}

	Delaunay dt( verts.begin(), verts.end() );

	/*
	// Write the triangulation to file
	vtkNew<vtkCellArray> sphereTri;
	for( auto fc = dt.all_faces_begin(); fc != dt.all_faces_end(); ++fc ){
	sphereTri->InsertNextCell(3);
	for( auto zz = 2; zz >= 0; --zz ){
	auto vid = dt.is_infinite( fc->vertex( zz ) )? 0 :
	fc->vertex(zz)->info();
	sphereTri->InsertCellPoint(vid);
	}
	}
	pointCloud->SetPolys( sphereTri );
	vtkNew<vtkPolyDataWriter> wr;
	wr->SetInputData( pointCloud );
	wr->SetFileName( "Sphere.vtk" );
	wr->Write();
	*/

	// Create the input scalar field f(theta,phi)
	std::complex<double_t> Qlm[NLM];
	for(auto pp = 0; pp < NLM; ++pp)
	    Qlm[pp] = 0.0;
	Qlm[ LM(shtns, 0, 0) ] = std::complex<double_t>(0,0);
	Qlm[ LM(shtns, 1, 0) ] = std::complex<double_t>(2,0);
	Qlm[ LM(shtns, 1, 1) ] = std::complex<double_t>(0,3);
	Qlm[ LM(shtns, 7, 7) ] = std::complex<double_t>(7,0);

	Eigen::VectorXd finput(N);
	for(auto i = 0; i < N; ++i){
	    Eigen::Vector3d q;
	    q = points.col(i);
	    double_t x = q(0);
	    double_t y = q(1);
	    double_t z = q(2);
	    auto phi = std::atan2(y,x);
	    finput(i) = SH_to_point(shtns, Qlm, z, phi);
	}

	//Now we need to first identify which triangles each of the quadrature
	//points belongs to. For this, we need the coordinates of the points
	Eigen::Matrix3Xd Qr(3,numQ);// Rotated quadrature points
	Eigen::Matrix3Xd Qsp(3,numQ);// StereoProjection of quadrature points


	// Write Quadrature points to VTK file
	//writeMatToVTK( Q0, "QuadPointsOrig.vtk" );

	// Now we need to rotate and project the quadrature points
	// p0 is already defined as the z = -1 plane
	// c is the point from which we are projecting (calculated earlier)
	Qr = rotMat*Q0; // Rotate the quadrature points

	// Write the rotated points to VTK
	//writeMatToVTK( Qr, "RotQuadPoints.vtk" );

	Eigen::Matrix3Xd lQ(3,numQ);
	lQ = (Qr.colwise() - c).colwise().normalized();//projn dirn unit vectors
	for( auto j=0; j < numQ; ++j ){
	    Qsp.col(j) = ((p0(2) - Qr(2,j))/lQ(2,j))*lQ.col(j) + Qr.col(j);
	}

	// Write Stereographic projections of quadrature points to VTK file
	//writeMatToVTK( Qsp, "ProjQuadPoints.vtk", true );

	//Finally let's try to locate these points in the Triangulation and
	//interpolate it as per the weights
	Eigen::Map<Eigen::VectorXd> fquad(f,numQ);
	auto interpolate = [&points, &Q0, &finput, &fquad](const size_t i,
		const size_t j, const size_t k, const size_t q){
	    Eigen::Vector3d v0 = points.col(i);
	    Eigen::Vector3d v1 = points.col(j);
	    Eigen::Vector3d v2 = points.col(k);
	    Eigen::Vector3d qp = Q0.col(q);
	    auto A0 = ((v1-qp).cross((v2-qp))).norm();
	    auto A1 = ((v2-qp).cross((v0-qp))).norm();
	    auto A2 = ((v0-qp).cross((v1-qp))).norm();
	    auto A = A0 + A1 + A2;
	    fquad(q) = (A0/A)*finput(i) + (A1/A)*finput(j) + (A2/A)*finput(k);
	};
	for( auto j=0; j < numQ; ++j ){
	    auto query = Point( Qsp(0,j), Qsp(1,j) );
	    Delaunay::Locate_type lt;
	    int li;
	    Face_handle face = dt.locate( query, lt, li );
	    switch(lt){
		case Delaunay::FACE:
		    {
			auto id0 = face->vertex(0)->info();
			auto id1 = face->vertex(1)->info();
			auto id2 = face->vertex(2)->info();
			interpolate( id0, id1, id2, j );
			break;
		    }
		case Delaunay::EDGE:
		    {
			auto id1 = face->vertex( (li + 1)%3 )->info();
			auto id2 = face->vertex( (li + 2)%3 )->info();
			Eigen::Vector3d v1, v2, qp;
			v1 = points.col(id1);
			v2 = points.col(id2);
			qp = Q0.col(j);
			double_t ratio = (qp - v1).norm()/(qp - v2).norm();
			fquad(j) = (finput(id1) + ratio*finput(id2))/(1 + ratio);
			break;
		    }
		case Delaunay::VERTEX:
		    fquad(j) = finput( face->vertex( li )->info() );
		    break;
		case Delaunay::OUTSIDE_CONVEX_HULL:
		    {
			Eigen::Vector3d v0, v1, v2;
			auto id0 = dt.is_infinite(face->vertex(0))?
			    0 : face->vertex(0)->info();
			auto id1 = dt.is_infinite(face->vertex(1))?
			    0 : face->vertex(1)->info();
			auto id2 = dt.is_infinite(face->vertex(2))?
			    0 : face->vertex(2)->info();
			interpolate( id0, id1, id2, j );
			break;
		    }
		default:
		    std::cout<< "Point " << j << " not found!" << std::endl;
	    }
	}

	/*
	   std::cout<< "Print interpolated values and actual values..."<<std::endl;
	   Eigen::VectorXd direct(numQ);
	   for(auto i = 0; i < numQ; ++i){
	   Eigen::Vector3d q;
	   q = Q0.col(i);
	   double_t x = q(0);
	   double_t y = q(1);
	   double_t z = q(2);
	   auto phi = std::atan2(y,x);
	   direct(i) = SH_to_point(shtns, Qlm, z, phi);
	   std::cout<< fquad(i) << " " << direct(i) << std::endl;
	   }
	   */

	// Now time for spherical harmonic analysis -- find the coefficients flm
	spat_to_SH(shtns, f, Slm);

	/*
	   std::cout<< "Printing the coefficients..." << std::endl;
	// Print the spherical harmonic coefficients
	for( auto l = 0; l <= lmax; ++l ){
	for( auto m = -l; m <= l; ++m ){
	std::complex<double_t> flm;
	if( m < 0){
	auto factor = (-m%2 == 0)? 1.0 : -1.0;
	flm = factor*std::conj(Slm[ LM(shtns,l,-m) ]);
	}
	else{
	flm = Slm[ LM(shtns,l,m) ];
	}
	std::cout<< l << " " << m << " " << flm << std::endl;
	}
	}
	*/
    }
    std::cout<< "Time for 1000 steps = " 
	<< ((float)(clock() - t))/CLOCKS_PER_SEC << std::endl;
    for( auto lm = 0; lm < NLM; ++lm){
	std::cout<< Slm[ lm ] << std::endl;
    }
    return 0;
}
