function [out] = hightlyNonlinearOscillator(parameters)
[c1,c2,m,r,t1,F1] = matsplit(parameters);
w0=sqrt((c1+c2)/m);
out=(3*r)-abs(((2*F1)/(m*w0^2))*sin(w0*t1/2));