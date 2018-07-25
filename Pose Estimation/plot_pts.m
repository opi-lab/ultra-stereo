clear, clc

load pts3D.mat
load stereoParams.mat

R = stereoParams.RotationOfCamera2;
t = stereoParams.TranslationOfCamera2;
or = R';
loc = -t * or;

figure, plotCamera('Location',[0 0 0],'Orientation',eye(3),'Size',20)
hold on, plotCamera('Location',loc,'Orientation',or,'Size',20)
plotCamera('Location',[0 0 0],'Orientation',eye(3),'Size',20)
plot3(X(1,:),X(2,:),X(3,:),'r*-'), plot3(X(4,:),X(5,:),X(6,:),'g*-')
plot3(X(7,:),X(8,:),X(9,:),'b*-'), grid, axis equal, axis manual

figure, plot3(X(1,:),X(3,:),-X(2,:),'r*-'), grid, hold on
plot3(X(4,:),X(6,:),-X(5,:),'g*-'), plot3(X(7,:),X(9,:),-X(8,:),'b*-')
xlabel('x'), ylabel('y'), zlabel('z')
