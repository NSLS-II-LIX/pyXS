#include "RQconv.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>


int zinger(int *data, int col, int row, int x, int y, int box, float tol){
    int ic,ir,cnt;
    float std2,avg,sum,sum2,v;

    // std = SUM{(x-avg)^2}/N = sum2/N - 2*sum/N*avg + avg^2 = sum2/N - avg^2

    cnt = sum = sum2 = 0;
    for (ir=y-box+1;ir<y+box;ir++) {
        if (ir < 0) continue;
        if (ir >= row) break;
        for (ic = x-box+1; ic < x+box; ic++) {
            if (ic < 0) continue;
            if (ic >= col) break;
            v = data[ic+ir*col];

            if (v > 0) {
                cnt++;
                sum+=v;
                sum2+=v*v;
            }
        }
    }

    if (!cnt) {
        return(0);
    }

    avg = sum/cnt;
    std2 = (sum2/cnt-avg*avg)/cnt;
    v = data[x+y*col]-avg;

    if (v*v > tol*tol*std2) {
        return(1);
    }

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

    c[0][0] = a[0][0]*b[0][0]+a[0][1]*b[1][0]+a[0][2]*b[2][0];
    c[0][1] = a[0][0]*b[0][1]+a[0][1]*b[1][1]+a[0][2]*b[2][1];
    c[0][2] = a[0][0]*b[0][2]+a[0][1]*b[1][2]+a[0][2]*b[2][2];

    c[1][0] = a[1][0]*b[0][0]+a[1][1]*b[1][0]+a[1][2]*b[2][0];
    c[1][1] = a[1][0]*b[0][1]+a[1][1]*b[1][1]+a[1][2]*b[2][1];
    c[1][2] = a[1][0]*b[0][2]+a[1][1]*b[1][2]+a[1][2]*b[2][2];

    c[2][0] = a[2][0]*b[0][0]+a[2][1]*b[1][0]+a[2][2]*b[2][0];
    c[2][1] = a[2][0]*b[0][1]+a[2][1]*b[1][1]+a[2][2]*b[2][1];
    c[2][2] = a[2][0]*b[0][2]+a[2][1]*b[1][2]+a[2][2]*b[2][2];

    for (i=0;i<3;i++) {
        for (j=0;j<3;j++) {
            b[i][j]=c[i][j];
        }
    }
    return;

}

void matrix_dot_vector(float (*m)[3],float *v) {
    float r[3];

    r[0] = m[0][0]*v[0]+m[0][1]*v[1]+m[0][2]*v[2];
    r[1] = m[1][0]*v[0]+m[1][1]*v[1]+m[1][2]*v[2];
    r[2] = m[2][0]*v[0]+m[2][1]*v[1]+m[2][2]*v[2];

    v[0] = r[0];
    v[1] = r[1];
    v[2] = r[2];
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

    for (i=0;i<3;i++) {
        for (j=0;j<3;j++) {
            exp->det_rot[i][j] = det_rot[i][j];
            exp->det_rot_T[i][j] = det_rot[j][i];
        }
    }
    return;
 
}

void dezinger(int *data, int row, int col, int *mask, int mrow, int mcol, int box, float tol) {
    // scattering data are int
    // mark up the mask as zingers are found
    int ic,ir;

    for (ir=0;ir<row;ir++) {
        for (ic=0;ic<col;ic++) {
            if (!mask[ic+ir*col]) {
                continue;
            }
            if (zinger(data,col,row,ic,ir,box,tol)) {
                mask[ic+ir*col]=0;
            }
        }
    }
}

vec2 qrqz2xy(int *data, int row, int col, ExpPara *exp, float qr, float qn, int flag) {
    // returns the pixel position (x,y) corresponding to (qr, qn)
    // when flag is non-zero, try to subtitue qr with qr0 (see below)

    vec2 ret;
    float alpha,theta,phi,tt;
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
            // will have to manually set q and phi, otherwise might have problem with qz calculation below
            q = sqrt(qr*qr+qn*qn);
        }
    }

    q = sqrt(qr*qr+qn*qn);
    theta = exp->wavelength*q/(4.0*Pi);
    if (theta>1.0) {
        return(ret);
    }
    theta = asin(theta);

    qz = q*sin(theta);
    qy = (qn-qz*sin(alpha))/cos(alpha);

    tt = q*q-qz*qz-qy*qy;
    if (fabs(tt) < 2.e-6) {
        tt = 0;
    }
    if (tt < 0) {
        if (flag) {
            printf("%.5f, %.5f, %.5f, %g\n", q, qy, qz, tt);
        }
        return(ret);
    }
    qx = sqrt(tt)*(qr<0?-1:1);

    phi = atan2(qy,qx);

    return (qphi2xy(data,row,col,exp,q,phi));
}

vec2 qphi2xy(int *data, int row, int col, ExpPara *exp, float q, float phi) {
    vec2 ret;
    float theta,rn,tt,dr,dz;
    float R13,R23,R33,v[3];

    ret.x=ret.y=-1.e10;

    theta = exp->wavelength*q/(4.0*Pi);
    if (theta>=1.0) {
        return(ret);
    }
    theta = asin(theta);

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

    return ret;
}

float pos2qrqz(int *data, int row, int col, ExpPara *exp, float x, float y, float *pqr, float *pqz, float *icor) {
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
    theta = 0.5*asin(tt/rr);
    q = 4.0*Pi*sin(theta)/exp->wavelength;
    dd2 = tt*tt+(dr-z)*(dr-z);
    dd = sqrt(dd2);

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
        v1[0] = v1[1] = 0;
        v1[2] = 1;
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
    if (!pqr || !pqz) {
        return(q);
    }

    phi = atan2(y,x);

    // sample normal is supposed to be vertical up (y-axis)
    // in reality it may be off by rn, clockwise is positive
    rn = exp->sample_normal*Pi/180.0;
    phi += rn;

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
    return (pos2qrqz(data, row, col, exp, x, y, 0, 0, &icor));
}

float xy2q_ic(int *data, int row, int col, ExpPara *exp, float x, float y, float *icor) {
    // this version will modify the scattering intensity
    // called by conv_to_iq
    //         q=xy2q_ic(data,row,col,exp,ic,ir)
    // should never be called twice for the same pixel
    float q;
    q = pos2qrqz(data, row, col, exp, x, y, 0, 0, icor);
    return q;
}

vec2 xy2qrqz(int *data, int row, int col, ExpPara *exp, float x, float y) {
    // this will return a (qr, qz) pair
    vec2 ret;
    float ic;
    ic=0;
    pos2qrqz(data, row, col, exp, x, y, &ret.x, &ret.y, &ic);
    return ret;
}

float find(double Udata[], double Ldata[], float v, int len, int *pidx) {
    // revised Aug 14, 2012
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
    int i, ic, ir, n;
    float q, v, v1, t, icor;
    float wt[1024] = {0}, wt2[1024] = {0}, ct[1024] = {0};
    double qUbound[1024], qLbound[1024];

    assert(len<1023);

    // this should correctly deal with non-uniformly spaced qgrid
    // the qgrid should be processed using slnXS.mod_grid()
    // each data point in q, q[i], should be located at the midpoint between qUbound[i] and qLbound[i]
    v = grid[1]-grid[0];
    for (i=0; i<len; i++) {
        if (i<len-1) {
            v = (grid[i+1]-grid[i])*2-v;
        }
        qLbound[i] = grid[i]-v/2;
        qUbound[i] = grid[i]+v/2;
    }

    for (ir=0; ir<nrow; ir++) {
        for (ic=0; ic<ncol; ic++) {
            v = (data[ic+ir*ncol]-1);    // added 1 in Data2D to avoid confusion with masked pixels
            if (v<0) {
                continue;         // behind mask
            }
            if (cor) {
                if (cor<0) {             // this is useful for building the IA corrrection into flat field
                    icor = -2.;
                    q = xy2q_ic(data,nrow,ncol,exp,ic,ir,&icor);
                    icor = 1./icor;
                } else {
                    icor = 2.;
                    q = xy2q_ic(data,nrow,ncol,exp,ic,ir,&icor);
                }
            } else {
                icor = 1.;
                q = xy2q(data,nrow,ncol,exp,ic,ir);
            }

            // revised Aug 14, 2012
            // intensity from each pixel goes into one single bin
            t = find(qUbound,qLbound,q,len,&n);
            if (t>0) { // do this only if q falls into a bin, #n
                v /= icor;
                wt[n] += v;
                wt2[n] += v*v;
                ct[n] += 1;
            }

        }
    }

    for (i=0; i<len; i++) {
        if (ct[i]>0) {
            v = wt[i]/ct[i];
            v1 = sqrt(fabs(wt2[i]/ct[i]-v*v))/sqrt(ct[i]);
            if (signbit(v1)) {
                printf("wt=%10.4f, wt2=%10.4f, ct=%10.4f, avg=%10.4f, sig=%10.4f\n",wt[i],wt2[i],ct[i],v,v1);
            }
        }
        else {
            v = v1 = 0;
        }
        grid[i+len] = v;
        grid[i+len*2] = v1;
    }

}

void pre_conv_Iqrqz(int *data, int nrow, int ncol, ExpPara *exp) {
    int ix, iy;
    float dq, qrmax, qrmin, qzmax, qzmin, qr, qz, tmp;

    // determine the best q-range for reciprocal space data
    for (ix=0; ix<=ncol; ix+=ncol/16) {
        for (iy=0; iy<=nrow; iy+=nrow/16) {
            tmp = 0;
            pos2qrqz(data, nrow, ncol, exp, ix, iy, &qr, &qz, &tmp);
            if (ix==0 && iy==0) {
                qrmin = qrmax = qr;
                qzmax = qzmin = qz;
            } else {
                if (qr<qrmin) {
                    qrmin = qr;
                }
                if (qr>qrmax) {
                    qrmax = qr;
                }
                if (qz<qzmin) {
                    qzmin = qz;
                }
                if (qz>qzmax) {
                    qzmax = qz;
                }
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
    exp->dq = dq=0.01*(int)(dq*100+0.5)*tmp;

    exp->nr = (int)((qrmax-qrmin)/dq)+1;
    exp->nz = (int)((qzmax-qzmin)/dq)+1;
    exp->qr0 = (int)(qrmin/dq)*dq;
    exp->qz0 = (int)(qzmin/dq)*dq;

}

float get_value(int *data, int nrow, int ncol, float fx, float fy) {
    int ix,iy;
    float t;

    ix = (int)fx;
    iy = (int)fy;
    if (ix<0 || iy<0 || ix>=ncol-1 || iy>=nrow-1) {
        return(0);
    }
    t = (1.-(fx-ix))*(1.-(fy-iy))*data[ix+iy*ncol];
    t += (fx-ix)*(1.-(fy-iy))*data[ix+(iy+1)*ncol];
    t += (1.-(fx-ix))*(fy-iy)*data[ix+1+iy*ncol];
    t += (fx-ix)*(fy-iy)*data[ix+1+(iy+1)*ncol];
    return t;

}

void conv_to_Iqrqz(int *data, int nrow, int ncol, int *dataout, int drow, int dcol, ExpPara *exp) {
    int ix,iy;
    vec2 t;

    for (ix=0; ix<dcol; ix++) {
        for (iy=0; iy<drow; iy++) {
            t = qrqz2xy(data,nrow,ncol,exp, exp->dq*ix+exp->qr0, exp->dq*(drow-1-iy)+exp->qz0, 0);
            dataout[ix+iy*dcol] = get_value(data, nrow, ncol, t.x, t.y);
        }
    }

}

void merge(int *data1, int nrow1, int ncol1, int *data2, int nrow2, int ncol2) {
    // this doesn't work
    // the original intention was to remove the zingers
    // but in reality it only picks the smaller value from data1 and data2
    // assume data1>data2, sum=data1+data2, diff=data1-data2, result=(sum-diff)/2=data2

    int i, j, sum, diff;

    if (nrow1!=nrow2 || ncol1!=ncol2) {
        return;
    }
    for (i=0; i<nrow1; i++) {
        for (j=0; j<=ncol1; j++) {
            sum = data1[i*ncol1+j]+data2[i*ncol1+j];
            diff = abs(data1[i*ncol1+j]-data2[i*ncol1+j]);
            data1[i*ncol1+j] = (sum-diff)/2;
        }
    }
}
