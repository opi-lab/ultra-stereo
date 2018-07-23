import cv2
import glob
import itertools
import numpy as np
import scipy.io as sio

def dst(l,p):
      if p.ndim == 1:
            p = np.array([p])
      
      return abs(l[0]*p[:,0]+l[1]*p[:,1]+l[2])/np.sqrt(l[0]**2+l[1]**2)


def detection(im,n):
      gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
      th,bw = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
      kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
      imo = cv2.morphologyEx(bw,cv2.MORPH_OPEN,kernel)
      
      _,contours,_ = cv2.findContours(imo,cv2.RETR_TREE,\
                                            cv2.CHAIN_APPROX_SIMPLE)
      areas = np.array([cv2.contourArea(c) for c in contours])
      perimeters = np.array([cv2.arcLength(c,1) for c in contours])
      R = 4*np.pi*areas/perimeters**2
      circ = np.array(contours)[R >= 0.7*R.max()]
      
      
      c = []
      for cont in circ:
            M = cv2.moments(cont)
            c.append([M['m10']/M['m00'],M['m01']/M['m00']])
      c = np.array(c)
      
      
      d = np.array([])
      for i in range(len(c)-1):
            d = np.append(d,np.linalg.norm(c[i]-c[i+1]))
      
      
      ind = np.argsort(d)[:n]
      c = c[ind]
      ring = circ[ind]
      
      for cnt in ring:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
      
      return im, c
    

def labeled(im,c):
      a = 0
      for pt1, pt2, pt3 in itertools.permutations(c,3):
            dot = np.dot(pt2-pt1,pt3-pt1)
            if a == 0:
                  dot_min = dot
                  org = pt1
                  p1 = pt2
                  p2 = pt3
                  
                  a = 1
                  continue
            if dot_min > dot:
                  dot_min = dot
                  org = pt1
                  p1 = pt2
                  p2 = pt3
      
      
      l1 = np.cross(np.append(org,1),np.append(p1,1))
      d1 = dst(l1,p2)[0]
      
      l2 = np.cross(np.append(org,1),np.append(p2,1))
      d2 = dst(l2,p1)[0]
      
      if d1 < d2:
            x = p2
            y = p1
      else:
            x = p1
            y = p2
      
      if (y[1] < org[1]) & (org[0] > x[0]):
            temp = org
            org = x
            x = temp
      elif (y[1] > org[1]) & (org[0] < x[0]):
            temp = org
            org = x
            x = temp
      
      font = cv2.FONT_HERSHEY_SIMPLEX
      cv2.putText(im,'0',tuple(np.int32(org)), font, 1,(0,0,255),3)
      cv2.putText(im,'x',tuple(np.int32(x)), font, 1,(0,0,255),3)
      cv2.putText(im,'y',tuple(np.int32(y)), font, 1,(0,0,255),3)
      
      return im, org, x, y


def draw(img, corners, imgpts):
    corner = tuple(np.int32(corners[0].ravel()))
    img = cv2.line(img,corner,tuple(np.int32(imgpts[0].ravel())), (255,0,0),5)
    img = cv2.line(img,corner,tuple(np.int32(imgpts[1].ravel())), (0,255,0),5)
    img = cv2.line(img,corner,tuple(np.int32(imgpts[2].ravel())), (0,0,255),5)
    return img

def drawCub(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img


cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detection', 640*2, 512)

I1 = glob.glob('Target/Camera1/*.bmp')
I2 = glob.glob('Target/Camera2/*.bmp')

Params = sio.loadmat('Params.mat')
K1 = Params['K1']
K2 = Params['K2']
R = Params['R']
t = Params['t']

P1 = K1 @ np.hstack([np.eye(3),np.zeros([3,1])])
P2 = K2 @ np.hstack([R,t])

#Pm = np.array([[0,0,0],[52.917,0,0],[0,84.667,0]])
Pm = np.array([[0,0,0],[52.917,0,0],[0,84.667,0],[52.917,84.667,0],
               [52.917/2,0,0],[52.917/2,84.667,0]])
#axis = 40*np.array([[1.,0,0], [0,1.,0], [0,0,-1.]])

axis = 20*np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3]])

pts3D = []
for im1, im2 in zip(I1,I2):
      im1 = cv2.imread(im1)
      im2 = cv2.imread(im2)
      
      im1, c1 = detection(im1,3)
      im2, c2 = detection(im2,3)
      
      im1, org1, x1, y1 = labeled(im1,c1)
      im2, org2, x2, y2 = labeled(im2,c2)
      
      p1 = np.array([org1,x1,y1]).T
      p2 = np.array([org2,x2,y2]).T
      
      X = cv2.triangulatePoints(P1,P2,p1,p2)
      X = X[:3]/X[-1]
      pts3D.append(X.T.flatten().tolist())
      
      xaxis = X[:,1]-X[:,0]
      yaxis = X[:,2]-X[:,0]
      zaxis = np.cross(xaxis,yaxis)
      p4th = org1 + (x1-org1) + (y1-org1)
      p5th = org1 + (x1-org1)/2
      p6th = org1 + (x1-org1)/2 + (y1-org1)
      p1 = np.array([org1,x1,y1,p4th,p5th,p6th]).T
      
#      ok, rvec, tvec = cv2.solvePnP(Pm, p1.T, K1, None)
      ok, rvec, tvec = cv2.solvePnP(Pm,p1.T,K1,None,np.array([[10],[-0.1],[-0.1]]),X[:,0].reshape(3,-1),1)
      ax, _ = cv2.projectPoints(axis, rvec, tvec, K1, None)
      
#      img = draw(im1.copy(),p1.T,ax)
      img = drawCub(im1.copy(),p1.T,ax)
      
      cv2.imshow('Detection',np.hstack([im1,img]))
      if cv2.waitKey(2000) & 0xFF == 27:
            break

#cv2.destroyAllWindows()
pts3D = np.array(pts3D).T
sio.savemat('pts3D.mat',{'X':pts3D})
#axisx = np.linalg.norm(X[:,0]-X[:,1])
#axisy = np.linalg.norm(X[:,0]-X[:,2])