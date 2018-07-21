clear, clc

load('stereoParams.mat')

K1 = stereoParams.CameraParameters1.IntrinsicMatrix';
K2 = stereoParams.CameraParameters2.IntrinsicMatrix';

R = stereoParams.RotationOfCamera2';
t = stereoParams.TranslationOfCamera2';


save Params.mat K1 K2 R t

P1 = K1*[eye(3) ones(3,1)];
P2 = K2*[R t];

org2 = [463.40037874 694.05805398];
org1 = [869.71484792 696.17461204];
Xorg = triangulate(org2,org1,stereoParams);
X2 = triangulation_pts([org2 1]',[org1 1]',P1,P2);

% X = cv2.triangulatePoints(P1,P2,np.array([463.40037874,694.05805398]),
% np.array([869.71484792,696.17461204]))
% (X[:3]/X[-1]).T

x2 = [686.00819214 717.61567562];
x1 = [1126.36839961 712.5709472];
Xx = triangulate(x2,x1,stereoParams);

y2 = [603.91314901 400.30434783];
y1 = [893.66646202 397.35085399];
Xy = triangulate(y2,y1,stereoParams);

ejex = norm(Xorg-Xx);
ejey = norm(Xorg-Xy);