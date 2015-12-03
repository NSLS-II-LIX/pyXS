#include "RQconv.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int zinger(int *data, int col, int row, int x, int y, int box, float tol){
  int ic,ir,cnt;
  float std2,avg,sum,sum2,v;

  // std = SUM{(x-avg)^2}/N = sum2/N - 2*sum/N*avg + avg^2 = sum2/N - avg^2 
  
  cnt=sum=sum2=0;
  for (ir=y-box+1;ir<y+box;ir++) {
    if (ir<0) continue;
    if (ir>=row) break;
    for (ic=x-box+1;ic<x+box;ic++) {
      if (ic<0) continue;
      if (ic>=col) break;
      v = data[ic+ir*col];
      //printf("%f\n",v);
      if (v>0) {
	cnt++;
	sum+=v;
	sum2+=v*v;
      }
    }
  }
  
  //printf("cnt=%d\n",cnt);
  if (!cnt) return(0);
  avg = sum/cnt; 
  std2 = (sum2/cnt-avg*avg)/cnt;
  v = data[x+y*col]-avg;
  
  //printf("std=%8.3f,  avg=%8.3f\n",sqrt(std2),avg);
  //printf("(%6d, %6d) = %8.1f\n",x,y,v+avg);

  // v = z - avg
  // the average excluding z should be 
  //    (avg*cnt-z)/(cnt-1) = (avg*cnt - (v+avg))/(cnt-1) = avg - v/(cnt-1) 
  //if (v*v>tol*tol*std2) return(avg-v/(cnt-1));
  if (v*v>tol*tol*std2) return(1);
  return 0;
}


void rot_matrix(float (*a)[3],char axis,float angle) {

  switch (axis) {
  case 'x':
  case 'X':
    a[0][0]=1;            a[0][1]=0;              a[0][2]=0;
    a[1][0]=0;            a[1][1]=cos(angle);     a[1][2]=-sin(angle);
    a[2][0]=0;            a[2][1]=sin(angle);     a[2][2]=cos(angle);    
    break;
  case 'y':
  case 'Y':
    a[0][0]=cos(angle);   a[0][1]=0;              a[0][2]=sin(angle);
    a[1][0]=0;            a[1][1]=1;              a[1][2]=0;
    a[2][0]=-sin(angle);  a[2][1]=0;              a[2][2]=cos(angle);    
    break;
  case 'z':
  case 'Z':
    a[0][0]=cos(angle);   a[0][1]=-sin(angle);    a[0][2]=0;
    a[1][0]=sin(angle);   a[1][1]= cos(angle);    a[1][2]=0;
    a[2][0]=0;            a[2][1]=0;              a[2][2]=1;    
    break;
  default:
    printf("invalid axis.");
    exit(1);
  }

  return;
}

void matrix_dot_matrix(float (*a)[3],float (*b)[3]) {
  float c[3][3];
  int i,j;
    
  c[0][0]=a[0][0]*b[0][0]+a[0][1]*b[1][0]+a[0][2]*b[2][0];
  c[0][1]=a[0][0]*b[0][1]+a[0][1]*b[1][1]+a[0][2]*b[2][1];
  c[0][2]=a[0][0]*b[0][2]+a[0][1]*b[1][2]+a[0][2]*b[2][2];

  c[1][0]=a[1][0]*b[0][0]+a[1][1]*b[1][0]+a[1][2]*b[2][0];
  c[1][1]=a[1][0]*b[0][1]+a[1][1]*b[1][1]+a[1][2]*b[2][1];
  c[1][2]=a[1][0]*b[0][2]+a[1][1]*b[1][2]+a[1][2]*b[2][2];

  c[2][0]=a[2][0]*b[0][0]+a[2][1]*b[1][0]+a[2][2]*b[2][0];
  c[2][1]=a[2][0]*b[0][1]+a[2][1]*b[1][1]+a[2][2]*b[2][1];
  c[2][2]=a[2][0]*b[0][2]+a[2][1]*b[1][2]+a[2][2]*b[2][2];

  for (i=0;i<3;i++) {
    for (j=0;j<3;j++) b[i][j]=c[i][j];
  }
  return;

}

void matrix_dot_vector(float (*m)[3],float *v) {
  float r[3];
  
  r[0]=m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2];
  r[1]=m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2];
  r[2]=m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2];

  v[0]=r[0]; v[1]=r[1]; v[2]=r[2];
  return;

}

void calc_rot(ExpPara *exp) {
  float dOrt,dTlt,dPhi,bTltX,bTltY;
  float tm[3][3],det_rot[3][3]={{1.,0,0},{0,1.,0},{0,0,1.}};
  int i,j;

  dOrt=exp->det_orient/180*Pi;
  dTlt=exp->det_tilt/180*Pi;
  dPhi=exp->det_phi/180*Pi;
  bTltX=exp->beam_tX/180*Pi;
  bTltY=exp->beam_tY/180*Pi;

  // tilt by dTlt along axis at dOrt from the y-axis
  rot_matrix(tm,'z',-dOrt);
  matrix_dot_matrix(tm,det_rot);
 
  rot_matrix(tm,'y',dTlt);
  matrix_dot_matrix(tm,det_rot);

  rot_matrix(tm,'z',dOrt);
  matrix_dot_matrix(tm,det_rot);
  
  rot_matrix(tm,'z',dPhi);
  matrix_dot_matrix(tm,det_rot);
 
    rot_matrix(tm,'x',bTltX);
  matrix_dot_matrix(tm,det_rot);

  rot_matrix(tm,'y',bTltY);
  matrix_dot_matrix(tm,det_rot);

  //printf("dOrt=%.2f, dTlt=%.2f, dPhi=%.2f, bTltX=%.2f, bTltY=%.2f\n",dOrt,dTlt,dPhi,bTltX,bTltY);

  for (i=0;i<3;i++) {
    for (j=0;j<3;j++) exp->det_rot[i][j] = det_rot[i][j];
    for (j=0;j<3;j++) exp->det_rot_T[i][j] = det_rot[j][i];
    //printf("%12.3f   %12.3f   %12.3f\n",det_rot[i][0],det_rot[i][1],det_rot[i][2]);  
  }
  return;
 
}

// scattering data are int  
// mark up the mask as zingers are found
void dezinger(int *data, int row, int col, int *mask, int mrow, int mcol, int box, float tol) {
  int ic,ir;

  for (ir=0;ir<row;ir++) {
    for (ic=0;ic<col;ic++) {
      if (!mask[ic+ir*col]) continue;
      if (zinger(data,col,row,ic,ir,box,tol)) mask[ic+ir*col]=0;
    }
  }

}

// returns the pixel position (x,y) corresponding to (qr, qn)
// when flag is non-zero, try to subtitue qr with qr0 (see below)
vec2 qrqz2xy(int *data, int row, int col, ExpPara *exp, float qr, float qn, int flag) {
  vec2 ret;
  float alpha,theta,phi,tt;
  //float q,qx2,qy2,qz2,qx1,qy1,qz1;
  float q,qx,qy,qz,qr0,k;

  ret.x=ret.y=-2e10;

  alpha = exp->incident_angle*Pi/180.0;

  // qr0 is the smallest allowed qr with a given qn
  // the Ewald sphere in the qr-qn plane is defined by a circle centered at 
  // k(cos(alpha),sin(alpha)), radius is k
  // (qr0/k - cos(alpha))^2 + (qn/k - sin(alpha))^2 = 1
  if (flag) {
    k = 2.0*Pi/exp->wavelength;
    tt = qn/k - sin(alpha);
    qr0 = fabs(sqrt(1.0 -tt*tt) - cos(alpha))*k;
    if (fabs(qr)<qr0) {
      qr=qr0*(qr>=0?1.:-1);
      //printf("qr0=%.5f,qr=%.5f,qn=%.5f,%.5f\n",qr0,qr,qn,
      //     sqrt((qr0/k-cos(alpha))*(qr0/k-cos(alpha))+(qn/k - sin(alpha))*(qn/k - sin(alpha))));
      // will have to manually set q and phi, otherwise might have problem with qz calculation below
      q = sqrt(qr*qr+qn*qn);
    }
  }

  q = sqrt(qr*qr+qn*qn);
  theta = exp->wavelength*q/(4.0*Pi);
  if (theta>1.0) return(ret);
  theta = asin(theta);

  qz = q*sin(theta);
  qy = (qn-qz*sin(alpha))/cos(alpha);

  tt = q*q-qz*qz-qy*qy;
  if (fabs(tt)<2.e-6) tt=0;
  if (tt<0) {
    if (flag) printf("%.5f, %.5f, %.5f, %g\n",q,qy,qz,tt);
    return(ret);
  }
  qx = sqrt(tt)*(qr<0?-1:1);

  phi = atan2(qy,qx);
  
  /*
  // q1: incident angle = 0
  // q2: with actual incident angle and tilt
  // note r is parallel to the sample, z is perpendicular to the sample
  // qz2 is one of the input parameter
  qz2 = qz;
  // qy1 (component along incident direction) is determined by q and theta, independent of phi 
  qy1 = -q*sin(theta);
  // qy1 = qy1 ( qy2, qz2, alpha)
  qy2 = (qy1+sin(alpha)*qz2)/cos(alpha);
  // now that qy2 and qz2 are known, qx2 can be calculated
  tt = q*q-qz2*qz2-qy2*qy2;
  if (tt<0) return(ret);
  qx2 = sqrt(tt)*(qr<0?-1:1);

  // now calculate qz1 and qx1 from q2 components
  qz1 = qy2*sin(alpha) + qz2*cos(alpha);
  qx1 = qx2;

  // really only need to figure out q and the the azimuthal angle 
  // (about the beam)
  phi = atan2(qz1,qx1);
  */

  return(qphi2xy(data,row,col,exp,q,phi));
}

vec2 qphi2xy(int *data, int row, int col, ExpPara *exp, float q, float phi) {
  vec2 ret;
  float theta,rn,tt,dr,dz;
  //float x1,y1,z1,x2,y2,z2,dx,dy,x,y,z,dOrt,dTlt,dPhi;
  float R13,R23,R33,v[3];

  ret.x=ret.y=-1.e10;

  theta = exp->wavelength*q/(4.0*Pi);
  if (theta>=1.0) return(ret);
  theta = asin(theta);

  //printf("row = %d, col = %d\n",row, col);
  //printf("q = %8.4f,  phi = %8.4f\n",q,phi);
  //printf("theta = %8.4f\n",theta);

  if (fabs(theta)<1e-6) { // beam center
    ret.x = exp->bm_ctr_x;
    ret.y = exp->bm_ctr_y;
    return ret;
  } 

  rn = exp->sample_normal*Pi/180.0;
  phi -= rn;
  
  R13 = exp->det_rot[0][2];
  R23 = exp->det_rot[1][2];
  R33 = exp->det_rot[2][2];
  
  tt = (R13*cos(phi)+R23*sin(phi))*tan(2.*theta);
  dr = exp->ratioDw*col;
  
  // pixel position in lab referece frame
  v[2] = dz = dr*tt/(tt-R33);
  v[0] = (dr-dz)*tan(2.0*theta)*cos(phi); 
  v[1] = (dr-dz)*tan(2.0*theta)*sin(phi); 
  
  // transform to detector frame
  matrix_dot_vector(exp->det_rot_T,v);
  
  // pixel position, note reversed y-axis
  ret.x = v[0] + exp->bm_ctr_x;
  ret.y = -v[1] + exp->bm_ctr_y;   

  /*
  dOrt = exp->det_orient/180*Pi;
  dTlt = exp->det_tilt/180*Pi;
  dPhi = exp->det_phi/180*Pi;
  
  // now determine the physical position of the pixel
  // non-trivial, see doc
  tt = tan(2.0*theta)*sin(dTlt)*(cos(dPhi)*cos(phi)+sin(dPhi)*sin(phi));
  dr = exp->ratioDw*col;
  dz = -dr*tt/(cos(dTlt)+tt);
  dx = (dz+dr)*tan(2.0*theta)*cos(phi); 
  dy = (dz+dr)*tan(2.0*theta)*sin(phi); 
  
  // now apply the inverse rotation matrices in reversed sequence
  // inverse of the roll
  // about z-axis, by detector_phi
  x1 = dx*cos(dPhi) + dy*sin(dPhi);
  y1 =-dx*sin(dPhi) + dy*cos(dPhi);
  z1 = dz;
  
  // inverse of tilt
  // about y-axis, by detector_tilt
  x2 = x1*cos(dTlt) - z1*sin(dTlt);
  y2 = y1;
  z2 = x1*sin(dTlt) + z1*cos(dTlt);
  
  // now rotate the detector back about z-axis so that pixel rows are horizontal
  x = x2*cos(dOrt) - y2*sin(dOrt);
  y = x2*sin(dOrt) + y2*cos(dOrt);
  // make sure z2 should be zero
  z = z2;
  
  ret.x = x + exp->bm_ctr_x;
  ret.y = -y + exp->bm_ctr_y;
  */

  return ret;
}

float pos2qrqz(int *data, int row, int col, ExpPara *exp, float x, float y, float *pqr, float *pqz, float *icor) {
  //float q,qx1,qy1,qz1,qx2,qy2,qz2;
  //float phi,theta,alpha,rn,dx,dy;
  //float x1,y1,z1,x2,y2,z2,z;
  //float dOrt,dTlt,dPhi;
  float q,qn,qr,qx,qy,qz;
  float phi,theta,alpha,rn,dr,dd,dd2,sinpsi2,cos_th;
  float z,tt,rr;
  float v[3],v1[3];
  int ncor;

  /* 
   *   the pixel rows in the WAXS CCD chip are not horizontal, but rotated about its center
   *   this angle is detector_orient
   *
   *   WAXS detector not only swings outward (or tilt), but also roll it upward
   *   detector_tilt is the angle by which the detector swings out/tilt (about y-axis)
   *   detector_phi is the angle by which the detector swings up (about z axis)
   *
   *   (cx, cy) is the nominal position of the direct beam on the detector
   *   this position does not change when the detector is rotated
   *
   *   detector_psi (detector swing) is retained for config file consistency
   */

  /*
  dOrt = exp->det_orient/180*Pi;
  dTlt = exp->det_tilt/180*Pi;
  dPhi = exp->det_phi/180*Pi;

  // NOTE: row number increase in the opposite direction of y-axis
  dx = x - exp->bm_ctr_x;
  dy = -(y - exp->bm_ctr_y);

  // in the image pixel rows are horizontal
  // detector_oreint = 45deg, when the WAXS detector is in dorizontal position
  x1 = dx*cos(dOrt) + dy*sin(dOrt);
  y1 = -dx*sin(dOrt) + dy*cos(dOrt);
  z1 = 0;

  // now swing out, really just tilting the detector by detector_tilt about y axis
  x2 = x1*cos(dTlt) + z1*sin(dTlt);
  y2 = y1;
  z2 = -x1*sin(dTlt) + z1*cos(dTlt);

  // now roll, about z-axis
  // definition of sign is opposite to detector_orient
  x = x2*cos(dPhi) - y2*sin(dPhi);
  y = x2*sin(dPhi) + y2*cos(dPhi);
  z = z2 + exp->ratioDw*col;

  x2 = sqrt(x*x+y*y);
  //theta = 0.5*atan2(fabs(z), x);
  theta = 0.5*atan(x2/fabs(z));
  //printf("theta = %g \n", theta/Pi*180); 
  q=4.0*Pi*sin(theta)/exp->wavelength;
  
  // pos_to_q() calls pos_to_qrz(x,y,0,0)
  if (!pqr || !pqz) return(q);

  dx = x;
  dy = y;

  // sample normal is supposed to be vertical up (y-axis)
  // the normal vector is rotated by rn, clockwise is positive 
  rn = exp->sample_normal*Pi/180.0;

  // phi is the angle between the projection of q on the detector and the x axis
  // y-axis is tilted by rn
  phi = atan2(dy,dx)+rn;
  //printf("phi = %7.3f\n",180/Pi*phi);

  // q components in the lab reference frame
  // y is in the direction of the incident beam
  // z is vertical up, x is therefore horizontal, pointing away
  qy1 = -sin(theta);
  qx1 = cos(theta)*cos(phi);
  qz1 = cos(theta)*sin(phi);

  // incident angle is alpha
  alpha = exp->incident_angle*Pi/180.0;
  qx2 =  qx1;
  qy2 =  qy1*cos(alpha) + qz1*sin(alpha);
  qz2 = -qy1*sin(alpha) + qz1*cos(alpha);

  //printf("(%6.3f, %6.3f, %6.3f) -> (%6.3f, %6.3f, %6.3f)\n",
  //	 qx1,qy1,qz1,qx2,qy2,qz2);

  *pqr = q*sqrt(qx2*qx2 + qy2*qy2)*(qx2<0?-1:1);
  *pqz = q*qz2;
  */

  v[0] = x - exp->bm_ctr_x;
  v[1] = -(y - exp->bm_ctr_y); 
  v[2] = 0;

  matrix_dot_vector(exp->det_rot,v);
  
  dr = exp->ratioDw*col;
  x = v[0];
  y = v[1];
  z = v[2]-dr;

  tt = sqrt(x*x+y*y);
  rr = sqrt(x*x+y*y+z*z);
  //theta = 0.5*atan(tt/(dr-z));
  theta = 0.5*asin(tt/rr);
  q = 4.0*Pi*sin(theta)/exp->wavelength;
  dd2 = tt*tt+(dr-z)*(dr-z);
  dd = sqrt(dd2);
  //printf("(%.1f,%.1f)=%f\n",x,y,q);  

  // code for corrections to be applied for each pixel:
  // 1: polarization only
  // 2: incident angle onto the detector only
  // other: 1 and 2
  ncor = (int)(*icor);

  if (*icor) {
    // intensity correction due to polarization factor
    // I(th) ~ I x cos(psi)
    // psi is the angle bwtween the incident polarization vector and the scattered beam
    // see Yang 2003 BJ
    // incident polarization vector is v1=(1, 0, 0)
    // scattered beam: v2 = (x, y, z) - (0, 0, dr)
    // cos(psi) = v1.v2/(|v1||v2|)
    
    sinpsi2 = (y*y+(dr-z)*(dr-z))/dd2;
    
    // intensity correction due to incident angle of X-ray onto the detector
    // Barba et.al., Rev. Sci. Inst., 70:2927, 1999
    // I(th) ~ I / ( cos(th)^2 x cos(th) )
    // th is the incident angle (from normal) of X-rays onto the detector
    // normal vector of the detector can be found by transforming vector (0, 0, 1)
    
    v1[0]=v1[1]=0; v1[2]=1;
    matrix_dot_vector(exp->det_rot,v1);
    
    cos_th = v1[0]*x + v1[1]*y + v1[2]*(z-dr);
    cos_th = fabs(cos_th)/dd;

    switch (ncor) {
    case 1: // polarization only
      *icor = sinpsi2;
      break;
    case 2: // incident angle onto the detector surface only
      *icor = cos_th*cos_th*cos_th;  // without polarization correction
      break;
    default:
      *icor = sinpsi2*cos_th*cos_th*cos_th;
    }
  }

  // pos_to_q() calls pos_to_qrz(x,y,0,0)
  if (!pqr || !pqz) return(q);

  phi = atan2(y,x);

  // sample normal is supposed to be vertical up (y-axis)
  // in reality it may be off by rn, clockwise is positive 
  rn = exp->sample_normal*Pi/180.0;
  phi += rn;
  //printf("phi = %7.2f \n", phi/Pi*180); 

  qz = q*sin(theta);
  qy = q*cos(theta)*sin(phi);  
  qx = q*cos(theta)*cos(phi);

  alpha = exp->incident_angle*Pi/180.0;
  qn = qy*cos(alpha)+qz*sin(alpha);
  qr = sqrt(q*q-qn*qn)*(qx<0?-1:1);

  *pqr = qr;
  *pqz = qn;
  return(q);

}

float xy2q(int *data, int row, int col, ExpPara *exp, float x, float y) {
  float icor;
  icor=0;
  return(pos2qrqz(data, row, col, exp, x, y, 0, 0, &icor));
}

// this version will modify the scattering intensity 
// called by conv_to_iq
//         q=xy2q_ic(data,row,col,exp,ic,ir)
// should never be called twice for the same pixel
float xy2q_ic(int *data, int row, int col, ExpPara *exp, float x, float y, float *icor) {
  float q;
  q=pos2qrqz(data, row, col, exp, x, y, 0, 0, icor);
  return q;
}

// this will return a (qr, qz) pair
vec2 xy2qrqz(int *data, int row, int col, ExpPara *exp, float x, float y) {
  vec2 ret;
  float ic;
  ic=0;
  pos2qrqz(data, row, col, exp, x, y, &ret.x, &ret.y, &ic);
  return ret;
}

/*** 
     old version before Aug 14, 2012

float find(double data[], float v, int len, int *pidx) {
  int p,p1,p2;

  //for (p=0;p<len;p++) printf("%f\t",data[p]);
  //printf("\n");

  p1=0; p2=len-1;
  if (v<=data[p1]) {*pidx=p1;return(1);}
  if (v>=data[p2]) {*pidx=p2;return(0);}
  while (p2-p1>1) {
    //printf("p1=%d   p2=%d\n",p1,p2);
    p=(p1+p2)/2;
    if (v>data[p]) p1=p;
    else if (v<data[p]) p2=p;
    else p1=p2=p;
  }
  
  *pidx = p1;

  //printf("%f is between %f and %f\n",v,data[p1],data[p1+1]);
  // fix this: what is data[] is not evenly spaced
  return (data[p1+1]-v)/(data[p1+1]-data[p1]);

} 

***/

// revised Aug 14, 2012

float find(double Udata[], double Ldata[], float v, int len, int *pidx) {
  int p,p1,p2;

  p1=0; p2=len-1;
  if (v<Ldata[p1] || v>Udata[p2]) return(-1); // out of bounds

  while (p2-p1>1) {
    //printf("p1=%d   p2=%d\n",p1,p2);
    p=(p1+p2)/2;
    if (v>Ldata[p]) p1=p;
    else if (v<Ldata[p]) p2=p;
    else p1=p2=p;
  }
  // v should be just above Ldata[p1]
  if (v>Udata[p1]) {
    //printf("[%.4f, %.4f], %.4f\n",Ldata[p1],Udata[p1],v);   
    return(-1);  // happen to fall between bins
  }
  *pidx = p1;
  return(1);

}

// corCode: 1=pol only, 2=IA only, other=both
void cor_IAdep_2D(int *data, int row, int col, ExpPara *exp, int corCode, int invert) {
  float q,cf;
  int ir,ic;
  for (ir=0;ir<row;ir++) {
    for (ic=0;ic<col;ic++) {
      cf=corCode;
      q=pos2qrqz(data, row, col, exp, ic, ir, 0, 0, &cf);
      if (invert) data[ic+ir*col]*=cf;
      else data[ic+ir*col]/=cf;
    }
  }
}

// convert 2D solution scattering/powder diffraction data into 1D curve 
void conv_to_Iq(int *data, int nrow, int ncol, ExpPara *exp, double *grid, int nc, int len, int cor) {
  int i,ic,ir,n;
  float q,v,v1,t,icor;
  //float *wt,*wt2,*ct;
  float wt[1024],wt2[1024],ct[1024];
  double qUbound[1024],qLbound[1024];

  assert(len<1023);
  //wt=(float *)malloc(sizeof(float)*len);
  //wt2=(float *)malloc(sizeof(float)*len);
  //ct=(float *)malloc(sizeof(float)*len);

  //printf("wt=%p, wt2=%p, ct=%p'\n",wt,wt2,ct);
  //printf("data=%p, exp=%p, grid=%p\n",data,exp,grid);
  
  for (i=0;i<len;i++) wt[i]=wt2[i]=ct[i]=0;
  //for (i=0;i<len-1;i++) {
  //  if (i==0 || i==len-2) qbound[i+1]=(grid[i]+grid[i+1])/2;
  //  else { // this is needed to deal with non-uniform grid
  //    v=grid[i]-grid[i-1];
  //    v1=grid[i+2]-grid[i+1];
  //    qbound[i+1]=(grid[i]*v+grid[i+1]*v1)/(v+v1);
  //  }
  //}
  //qbound[0]=0;
  //qbound[i]=grid[i];

  // this should correctly deal with non-uniformly spaced qgrid
  // the qgrid should be processed using slnXS.mod_grid()
  // each data point in q, q[i], should be located at the midpoint between qUbound[i] and qLbound[i]
  v=grid[1]-grid[0];
  for (i=0;i<len;i++) {
    if (i<len-1) v=(grid[i+1]-grid[i])*2-v;
    qLbound[i]=grid[i]-v/2;
    qUbound[i]=grid[i]+v/2;
  }

  //for (i=0;i<len;i++) printf("%.4f   %.4f\n",qLbound[i],qUbound[i]);
  //printf("*******\n");

  for (ir=0;ir<nrow;ir++) {
    for (ic=0;ic<ncol;ic++) {
      v=(data[ic+ir*ncol]-1);    // added 1 in Data2D to avoid confusion with masked pixels
      if (v<0) continue;         // behind mask
      if (cor) {
	if (cor<0) {             // this is useful for building the IA corrrection into flat field
	  icor=-2.;
	  q=xy2q_ic(data,nrow,ncol,exp,ic,ir,&icor);
	  icor=1./icor;
	} else {
	  icor=2.;
	  q=xy2q_ic(data,nrow,ncol,exp,ic,ir,&icor);
	}
      } else {
	icor=1.;
	q=xy2q(data,nrow,ncol,exp,ic,ir);
      }

      /****
	   code before Aug 14, 2012
	   try to split intensity from each pixel into two bins
	   *****	   
      //t=find(grid,q,len,&n); // find where to put the intensity
      if (t<0 || t>1) printf("error with find(), t=%f\n",t);
      v/=icor;   
      if (v<0) printf("found negative intensity, v=%f\n",v);
      wt[n]+=v*t;
      wt[n+1]+=v*(1.-t);
      wt2[n]+=v*v*t;
      wt2[n+1]+=v*v*(1.-t);
      ct[n]+=t;
      ct[n+1]+=1.-t;
      ***/

      // revised Aug 14, 2012
      // intensity from each pixel goes into one single bin
      t=find(qUbound,qLbound,q,len,&n);
      if (t>0) { // do this only if q falls into a bin, #n
	v/=icor;   
	wt[n]+=v;
	wt2[n]+=v*v;    
	ct[n]+=1;
      }
	
      //if (v>100 && nrow!=ncol) {
      //printf("(c,r)=(%d,%d): \n",ic,ir);
      //printf("\tn=%d, \tv=%7.1f, t=%7.4f, wt=%10.4f, wt2=%10.4f, ct=%10.4f\n",n,v,t,wt[n],wt2[n],ct[n]);
      //printf("\tn+1=%d, \tv=%7.1f, t=%7.4f, wt=%10.4f, wt2=%10.4f, ct=%10.4f\n",n+1,v,1.-t,wt[n+1],wt2[n+1],ct[n+1]);
      //
      // if (nrow!=ncol && (n==148 || n==147)) {
      //	printf("%9g    %9g    %9g\n",ct[148],wt[148],wt2[148]);
      //}
    }
  }

  //if (nrow!=ncol) {
  //	printf("%9g    %9g    %9g\n",ct[148],wt[148],wt2[148]);
  //  for (i=0;i<len;i++) printf("%9g    %9g    %9g\n",ct[i],wt[i],wt2[i]);
  //}

  for (i=0;i<len;i++) {
    //if (len<150) {printf("%7.4f, %7.4f\n",grid[i],);
    if (ct[i]>0) {
      v=wt[i]/ct[i];
      v1=sqrt(fabs(wt2[i]/ct[i]-v*v))/sqrt(ct[i]);
      if (signbit(v1)) printf("wt=%10.4f, wt2=%10.4f, ct=%10.4f, avg=%10.4f, sig=%10.4f\n",wt[i],wt2[i],ct[i],v,v1);
    }
    else v=v1=0;
    grid[i+len]=v;
    grid[i+len*2]=v1;
    //printf("%7.4f    %7.4f\n",grid[i],grid[i+len]);
  }

  //free(ct);
  //free(wt2);
  //free(wt);
}

void pre_conv_Iqrqz(int *data, int nrow, int ncol, ExpPara *exp) {
  int ix,iy;
  float dq,qrmax,qrmin,qzmax,qzmin,qr,qz,tmp;

  // determine the best q-range for reciprocal space data
  for (ix=0;ix<=ncol;ix+=ncol/16) {
    for (iy=0;iy<=nrow;iy+=nrow/16) {
      tmp=0;
      pos2qrqz(data,nrow,ncol,exp,ix,iy,&qr,&qz,&tmp);
      if (ix==0 && iy==0) {
	qrmin=qrmax=qr;
	qzmax=qzmin=qz;
      } else {
	if (qr<qrmin) qrmin=qr;
	if (qr>qrmax) qrmax=qr;
	if (qz<qzmin) qzmin=qz;
	if (qz>qzmax) qzmax=qz;
      }
    }
  }
  
  dq = qrmax-qrmin;
  tmp = qzmax-qzmin;
  if (tmp>dq) dq=tmp;

  if (nrow>ncol) dq/=nrow;
  else dq/=ncol;
  
  tmp=1;
  while (dq>1) {tmp*=10;dq/=10;}
  while (dq<0.1) {tmp/=10;dq*=10;}
  exp->dq=dq=0.01*(int)(dq*100+0.5)*tmp;
  
  exp->nr=(int)((qrmax-qrmin)/dq)+1;  
  exp->nz=(int)((qzmax-qzmin)/dq)+1;
  exp->qr0=(int)(qrmin/dq)*dq;
  exp->qz0=(int)(qzmin/dq)*dq;

}

float get_value(int *data, int nrow, int ncol, float fx, float fy) {
  int ix,iy;
  float t;

  ix = (int)fx;
  iy = (int)fy;
  if (ix<0 || iy<0 || ix>=ncol-1 || iy>=nrow-1) return(0);
  t = (1.-(fx-ix))*(1.-(fy-iy))*data[ix+iy*ncol];
  t += (fx-ix)*(1.-(fy-iy))*data[ix+(iy+1)*ncol];
  t += (1.-(fx-ix))*(fy-iy)*data[ix+1+iy*ncol];
  t += (fx-ix)*(fy-iy)*data[ix+1+(iy+1)*ncol];
  return t;
  
}

//
void conv_to_Iqrqz(int *data, int nrow, int ncol, 
		   int *dataout, int drow, int dcol, 
		   ExpPara *exp) {
  int ix,iy;
  vec2 t;

  for (ix=0;ix<dcol;ix++) {
    for (iy=0;iy<drow;iy++) {
	  t = qrqz2xy(data,nrow,ncol,exp,
		      exp->dq*ix+exp->qr0, exp->dq*(drow-1-iy)+exp->qz0, 0); 
	  dataout[ix+iy*dcol] = get_value(data, nrow, ncol, t.x, t.y);
    }
  }
  
}

// this doesn't work
// the original intention was to remove the zingers
// but in reality it only picks the smaller value from data1 and data2
// assume data1>data2, sum=data1+data2, diff=data1-data2, result=(sum-diff)/2=data2
void merge(int *data1, int nrow1, int ncol1, int *data2, int nrow2, int ncol2) {
  int i, j, sum, diff;
  
  if (nrow1!=nrow2 || ncol1!=ncol2) return;
  for (i=0; i<nrow1; i++) {
    for (j=0;j<=ncol1; j++) {
      sum=data1[i*ncol1+j]+data2[i*ncol1+j];
      diff=fabs(data1[i*ncol1+j]-data2[i*ncol1+j]);
      data1[i*ncol1+j]=(sum-diff)/2;
    }
  }
}
