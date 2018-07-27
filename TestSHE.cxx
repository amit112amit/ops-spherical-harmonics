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
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include "SHTools.h"

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

    // Test function: 2*Y_1,-1 + Y_10 + 3*Y_11, so l=1, m=-1,0,1
    int lmax = std::floor( sqrt(N) - 1 );
    Eigen::VectorXd plms( (lmax + 1)*(lmax + 2)/2 );
    auto Plm = [&plms](const size_t l, const size_t m){
	return plms( l*(l+1)/2 + m );
    };

    // Map the point coordinates into a matrix
    Map3Xd pointsOrig((double_t*)pointCloud->GetPoints()
	    ->GetData()->GetVoidPointer(0),3,N);
    Matrix3Xd points(3,N);
    points = pointsOrig;

    //**********************************************************************//
    // Create the Gauss-Lengendre grid
    int nlat, nlong;
    Eigen::VectorXd latglq(lmax + 1);
    Eigen::VectorXd longlq(2*lmax + 1);
    glqgridcoord_wrapper_(latglq.data(), longlq.data(), &lmax, &nlat, &nlong);
    auto numQ = nlat*nlong;

    Eigen::VectorXd gridglq((lmax + 1)*(2*lmax + 1));
    Eigen::VectorXd plx( (lmax + 1)*(lmax + 1)*(lmax + 2)/2 );
    Eigen::VectorXd w( lmax + 1 ), zero( lmax + 1 );
    Eigen::VectorXd cilm( 2*(lmax + 1)*(lmax + 1) );
    Eigen::VectorXd pspectrum( lmax + 1 );

    // Pre-compute all the matrices needed for expansion
    shglq_wrapper_(&lmax, zero.data(), w.data(), plx.data() );

    //Now we need to first identify which triangles each of the quadrature
    //points belongs to. For this, we need the coordinates of the points
    Eigen::Matrix3Xd Q0(3,numQ);// Original quadrature points
    Eigen::Matrix3Xd Qr(3,numQ);// Rotated quadrature points
    Eigen::Matrix3Xd Qsp(3,numQ);// StereoProjection of quadrature points

    size_t index = 0;
    for( auto ip = 0; ip < nlong; ++ip){
	auto phi = longlq(ip)*M_PI/180.0;
	auto sin_p = std::sin(phi);
	auto cos_p = std::cos(phi);
	for( auto it = 0; it < nlat; ++it){
	    auto theta = (90.0 - latglq(it))*M_PI/180.0;
	    auto sin_t = std::sin(theta);
	    Q0.col(index++) << sin_t*cos_p, sin_t*sin_p, std::cos(theta);
	}
    }

    // Write Quadrature points to VTK file
    //writeMatToVTK( Q0, "QuadPointsOrig.vtk" );

    //**********************************************************************//

    VectorXd finput(N);
    // A function to interpolate data to the quadrature points
    auto interpolate = [&points, &Q0, &gridglq, &finput](const size_t i,
	    const size_t j, const size_t k, const size_t q){
	Eigen::Vector3d v0 = points.col(i);
	Eigen::Vector3d v1 = points.col(j);
	Eigen::Vector3d v2 = points.col(k);
	Eigen::Vector3d qp = Q0.col(q);
	auto A0 = ((v1-qp).cross((v2-qp))).norm();
	auto A1 = ((v2-qp).cross((v0-qp))).norm();
	auto A2 = ((v0-qp).cross((v1-qp))).norm();
	auto A = A0 + A1 + A2;
	gridglq(q) = (A0/A)*finput(i) + (A1/A)*finput(j) + (A2/A)*finput(k);
    };

    clock_t t = clock();
    for( auto z = 0; z < 1000; ++z){

	// Project points to unit sphere
	points.colwise().normalize();

	// Reset the center of the sphere to origin by translating
	Vector3d center = points.rowwise().mean();
	points = points.colwise() - center;

	// Calculate the function to be expanded using Spherical Harmonics
	for(auto i = 0; i < N; ++i){
	    Eigen::Vector3d q;
	    q =  points.col(i);
	    double_t x = q(0);
	    double_t y = q(1);
	    double_t z = q(2);
	    plmon_wrapper_(plms.data(), &lmax, &z);
	    auto phi = std::atan2(y,x);
	    finput(i) = 2*Plm(1,1)*std::sin(phi) + Plm(1,0) +
		3*Plm(1,1)*std::cos(phi);
	}

	// Rotate all points so that the 0th point is along z-axis
	Vector3d c = points.col(0);
	double_t cos_t = c(2);
	double_t sin_t = std::sqrt( 1 - cos_t*cos_t );
	Vector3d axis;
	axis << c(1), -c(0), 0.;
	axis.normalize();
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
	l = (l0.colwise() - c).colwise().normalized(); // dirns of projections
	for( auto j=0; j < N-1; ++j ){
	    proj.col(j) = ((p0(2) - l0(2,j))/l(2,j))*l.col(j) + l0.col(j);
	}

	// Write the rotated points to VTK
	//writeMatToVTK( proj, "ProjBasePoints.vtk" );

	// Insert the projected points in a CGAL vertex_with_info vector
	std::vector< std::pair< Point, unsigned> > verts;
	for( auto j=0; j < N-1; ++j ){
	    verts.push_back(std::make_pair(Point(proj(0,j),proj(1,j)),j+1));
	}

	// Triangulate
	Delaunay dt( verts.begin(), verts.end() );

	/*
	// Write the triangulation to file
	vtkNew<vtkPolyData> sphere;
	vtkNew<vtkCellArray> sphereTri;
	vtkNew<vtkPoints> spherePts;
	for(auto zz = 0; zz < N; ++zz){
	Vector3d pp = points.col(zz);
	spherePts->InsertNextPoint( &pp(0) );
	}
	for( auto fc = dt.all_faces_begin(); fc != dt.all_faces_end(); ++fc ){
	sphereTri->InsertNextCell(3);
	for( auto zz = 2; zz >= 0; --zz ){
	auto vid = dt.is_infinite( fc->vertex( zz ) )? 0 :
	fc->vertex(zz)->info();
	sphereTri->InsertCellPoint(vid);
	}
	}
	sphere->SetPoints( spherePts );
	sphere->SetPolys( sphereTri );
	vtkNew<vtkPolyDataWriter> wr;
	wr->SetInputData( sphere );
	wr->SetFileName( "UnitSphere.vtk" );
	wr->Write();
	*/

	// Rotate and project the quadrature points
	Qr = rotMat*Q0; // Rotate the quadrature points

	// Write the rotated points to VTK
	//writeMatToVTK( Qr, "RotQuadPoints.vtk" );

	Eigen::Matrix3Xd lQ(3,numQ);
	lQ = (Qr.colwise() - c).colwise().normalized();//projn dirn unit vectors
	for( auto j=0; j < numQ; ++j ){
	    Qsp.col(j) = ((p0(2) - Qr(2,j))/lQ(2,j))*lQ.col(j) + Qr.col(j);
	}

	// Write Stereographic projections of quadrature points to VTK file
	//writeMatToVTK( Qsp, "ProjQuadPoints.vtk" );

	// Locate the quadrature points using stereographic triangulation
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
			gridglq(j) = (finput(id1) +
				ratio*finput(id2))/(1 + ratio);
			break;
		    }
		case Delaunay::VERTEX:
		    gridglq(j) = finput( face->vertex( li )->info() );
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
		    std::cout<< "Quadrature point " << j << " not found!"
			<< std::endl;
	    }
	}

	// Expand using Gauss-Legendre Quadrature and get the power spectrum
	shexpandglq_wrapper_( cilm.data(), &lmax, gridglq.data(), w.data(),
		plx.data());
	shpowerspectrum_wrapper_( cilm.data(), &lmax, pspectrum.data() );
    }
    t = clock() - t;
    std::cout<< "Time for 1000 steps GLQ = " << ((float)t)/CLOCKS_PER_SEC
	<< std::endl;

    // Cross-check
    Eigen::VectorXd gridglq_out( (lmax + 1)*(2*lmax + 1) );
    makegridglq_wrapper_(gridglq_out.data(), cilm.data(), &lmax, plx.data());
    std::cout<< "Error norm of GLQ = " << (gridglq - gridglq_out).norm()
	<< std::endl;

    // Print the coefficients
    std::cout<< "l\tm\tAlm_GLQ" << std::endl;
    for( int l = 0; l <= lmax; ++l ){
	for( int m = -l; m <=l; ++m ){
	    int i = m < 0? 1 : 0;
	    int n = m < 0? -m: m;
	    std::cout<< l << "\t" << m << "\t"
		<< cilm( i + 2*(l + n*(lmax + 1)) )<< "\t"
		<< std::endl;
	}
    }
    return 0;
}
