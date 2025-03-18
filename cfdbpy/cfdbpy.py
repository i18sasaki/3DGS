import numpy as np
import bpy
DELTAT=0.2#Δt
OMEGA=1.8#SORの加速係数
Re=2300#レイノルズ数
maxframe=400

def sign(f):
    if f < 0:
        return -1
    return 1

#時間微分項と移流項。壁のところは速度を更新してないことに注意
def Advection(x,y,z,vx,vy,vz):
    vx_after = np.zeros((x+3,y+2,z+2), dtype=np.float64)  # x速度
    vy_after = np.zeros((x+2,y+3,z+2), dtype=np.float64)  # y速度
    vz_after = np.zeros((x+2,y+2,z+3), dtype=np.float64)  # z速度
    #x方向の移流
    for i in range(2,x+1):
        for j in range(1,y+1):
            for k in range(1,z+1):
                u = vx[i,j,k]
                v = (vy[i-1,j,k] + vy[i,j,k] + vy[i-1,j+1,k] + vy[i,j+1,k]) / 4
                w = (vz[i-1,j,k] + vz[i,j,k] + vz[i-1,j,k+1] + vz[i,j,k+1]) / 4
                vx_after[i,j,k] = vx[i,j,k] - u*sign(u)*(vx[i,j,k] - vx[i+sign(u),j,k])*DELTAT - v*sign(v)*(vx[i,j,k] - vx[i,j+sign(v),k])*DELTAT - w*sign(w)*(vx[i,j,k] - vx[i,j,k+sign(w)])*DELTAT
    #y方向の移流
    for i in range(1,x+1):
        for j in range(2,y+1):
            for k in range(1,z+1):
                u = (vx[i,j-1,k] + vx[i,j,k] + vx[i+1,j-1,k] + vx[i+1,j,k]) / 4
                v = vy[i,j,k]
                w = (vz[i,j-1,k] + vz[i,j,k] + vz[i,j-1,k+1] + vz[i,j,k+1]) / 4
                vy_after[i,j,k] = vy[i,j,k] - u*sign(u)*(vy[i,j,k] - vy[i+sign(u),j,k])*DELTAT - v*sign(v)*(vy[i,j,k] - vy[i,j+sign(v),k])*DELTAT - w*sign(w)*(vy[i,j,k] - vy[i,j,k-sign(w)])*DELTAT
    #z方向の移流
    for i in range(1,x+1):
        for j in range(1,y+1):
            for k in range(2,z+1):
                u = (vx[i,j,k-1] + vx[i,j,k] + vx[i+1,j,k-1] + vz[i+1,j,k]) / 4
                v = (vy[i,j,k-1] + vy[i,j,k] + vy[i,j+1,k-1] + vy[i,j+1,k]) / 4
                w = vz[i,j,k]
                vz_after[i,j,k] = vz[i,j,k] - u*sign(u)*(vz[i,j,k] - vz[i+sign(u),j,k])*DELTAT - v*sign(v)*(vz[i,j,k] - vz[i,j+sign(v),k])*DELTAT - w*sign(w)*(vz[i,j,k] - vz[i,j,k+sign(w)])*DELTAT
    vx[:,:,:] = vx_after.copy()
    vy[:,:,:] = vy_after.copy()
    vz[:,:,:] = vz_after.copy()
    return

#粘性の計算
def Viscosity(x,y,z,vx,vy,vz):
    vx_after = np.zeros((x+3,y+2,z+2), dtype=np.float64)  # x速度
    vy_after = np.zeros((x+2,y+3,z+2), dtype=np.float64)  # y速度
    vz_after = np.zeros((x+2,y+2,z+3), dtype=np.float64)  # z速度
    for i in range(1,x+1):
        for j in range(1,y+1):
            for k in range(1,z+1):
                vx_after[i,j,k]=vx[i,j,k]+1/Re*(vx[i+1,j,k]+vx[i-1,j,k]+vx[i,j+1,k]+vx[i,j-1,k]+vx[i,j,k+1]+vx[i,j,k-1]-6*vx[i,j,k])*DELTAT
                vy_after[i,j,k]=vy[i,j,k]+1/Re*(vy[i+1,j,k]+vy[i-1,j,k]+vy[i,j+1,k]+vy[i,j-1,k]+vy[i,j,k+1]+vy[i,j,k-1]-6*vy[i,j,k])*DELTAT
                vz_after[i,j,k]=vz[i,j,k]+1/Re*(vz[i+1,j,k]+vz[i-1,j,k]+vz[i,j+1,k]+vz[i,j-1,k]+vz[i,j,k+1]+vz[i,j,k-1]-6*vz[i,j,k])*DELTAT
    vx[:,:,:] = vx_after.copy()
    vy[:,:,:] = vy_after.copy()
    vz[:,:,:] = vz_after.copy()
    return 

#発散の計算
def Div(x,y,z,vx,vy,vz,s):
    for i in range(1,x+1):
        for j in range(1,y+1):
            for k in range(1,z+1):
                s[i,j,k]=(-vx[i,j,k] -vy[i,j,k] -vz[i,j,k] +vx[i+1,j,k] +vy[i,j+1,k] +vz[i,j,k+1])/DELTAT
    return

#圧力項の計算SOR法
def Poisson(x,y,z,p,s):
    eps = 1.0e-2
    while True:
        delta=0
        delta_new=0
        for i in range(1,x+1):
            for j in range(1,y+1):
                for k in range(1,z+1):
                    #もし壁なら、p[i,j,k]の圧力を代入。壁情報をbool型配列で管理しておくといろんな壁が再現できる
                    if i==1:#左の壁
                        p[i-1,j,k]=p[i,j,k]
                    if i==x:#右の壁
                        p[i+1,j,k]=p[i,j,k]
                    if j==1:#上の壁
                        p[i,j-1,k]=p[i,j,k]
                    if j==y:#下の壁
                        p[i,j+1,k]=p[i,j,k]
                    if k==1:
                        p[i,j,k-1]=p[i,j,k]
                    if k==z:
                        p[i,j,k+1]=p[i,j,k]
                    old_p = p[i,j,k]
                    p[i,j,k] = (1.0 - OMEGA) * p[i,j,k] + OMEGA / 6 * (p[i-1,j,k] + p[i+1,j,k] + p[i,j-1,k] + p[i,j+1,k] +p[i,j,k-1] +p[i,j,k+1] - (s[i,j,k]))
                    delta+=abs(p[i,j,k]-old_p)
                    delta_new+=abs(p[i,j,k])
        if  delta < eps*delta_new :
            break
    return

#圧力項による速度の修正
def Rhs(x,y,z,vx,vy,vz,p):
    for i in range(1,x+1):
        for j in range(1,y+1):
            for k in range(1,z+1):
                vx[i,j,k] -= (p[i,j,k] - p[i-1,j,k]) * DELTAT
                vy[i,j,k] -= (p[i,j,k] - p[i,j-1,k]) * DELTAT
                vz[i,j,k] -= (p[i,j,k] - p[i,j,k-1]) * DELTAT
    return

#粒子座標の速度を抽出して座標更新
#スタガード格子なのでxとy速度場の参照がずれる
def Flowparticles(vx,vy,vz,prt):
    for i in range(prt.shape[0]):
        xx=np.clip(prt[i,0],0.0,vx.shape[0]-2)
        yy=np.clip(prt[i,1],0.0,vy.shape[1]-2)
        zz=np.clip(prt[i,2],0.0,vz.shape[2]-2)
        ixx = np.int32(xx)
        iyy = np.int32(yy-0.5)
        izz = np.int32(zz-0.5)
        sxx = xx - ixx
        syy = (yy-0.5) - iyy
        szz = (zz-0.5) - izz
        spdx = (((1.0 - sxx) * vx[ixx,iyy,izz] + sxx * vx[ixx+1,iyy,izz]) * (1.0 - syy) + (
                    (1.0 - sxx) * vx[ixx,iyy+1,izz] + sxx * vx[ixx+1,iyy+1,izz]) * syy) * (1.0 - szz) + (
                ((1.0 - sxx) * vx[ixx,iyy,izz+1] + sxx * vx[ixx+1,iyy,izz+1]) * (1.0 - syy) + (
                    (1.0 - sxx) * vx[ixx,iyy+1,izz+1] + sxx * vx[ixx+1,iyy+1,izz+1]) * syy) * szz * DELTAT
        ixx = np.int32(xx-0.5)
        iyy = np.int32(yy)
        sxx = (xx-0.5) - ixx
        syy = yy - iyy
        spdy = (((1.0 - sxx) * vy[ixx,iyy,izz] + sxx * vy[ixx+1,iyy,izz]) * (1.0 - syy) + (
                    (1.0 - sxx) * vy[ixx,iyy+1,izz] + sxx * vy[ixx+1,iyy+1,izz]) * syy) * (1.0 - szz) + (
                ((1.0 - sxx) * vy[ixx,iyy,izz+1] + sxx * vy[ixx+1,iyy,izz+1]) * (1.0 - syy) + (
                    (1.0 - sxx) * vy[ixx,iyy+1,izz+1] + sxx * vy[ixx+1,iyy+1,izz+1]) * syy) * szz * DELTAT
        iyy = np.int32(yy-0.5)
        izz = np.int32(zz)
        syy = (yy-0.5) - iyy
        szz = zz - izz
        spdz = (((1.0 - sxx) * vy[ixx,iyy,izz] + sxx * vy[ixx+1,iyy,izz]) * (1.0 - syy) + (
                    (1.0 - sxx) * vy[ixx,iyy+1,izz] + sxx * vy[ixx+1,iyy+1,izz]) * syy) * (1.0 - szz) + (
                ((1.0 - sxx) * vy[ixx,iyy,izz+1] + sxx * vy[ixx+1,iyy,izz+1]) * (1.0 - syy) + (
                    (1.0 - sxx) * vy[ixx,iyy+1,izz+1] + sxx * vy[ixx+1,iyy+1,izz+1]) * syy) * szz * DELTAT
        prt[i, 0] += spdx
        prt[i, 1] += spdy
        prt[i, 2] += spdz
    return

def makeParticle(prt):
    verts = [(0,0,0),(0,0.25,0),(0.25,0.25,0),(0.25,0,0),(0,0,0.25),(0,0.25,0.25),(0.25,0.25,0.25),(0.25,0,0.25)]
    faces = [(0,1,2,3), (7,6,5,4), (0,4,5,1), (1,5,6,2), (2,6,7,3), (3,7,4,0)]
    mesh = bpy.data.meshes.new("Cube_mesh")
    mesh.from_pydata(verts,[],faces)
    mesh.update(calc_edges=True)
    for index,r in enumerate(prt):
        print(index)
        obj = bpy.data.objects.new(str(index), mesh)
        obj.location = r
        #サブディビジョンサーフェスを適応する
        obj.modifiers.new("subd", type='SUBSURF')
        #サブディビジョンサーフェスのレベルを決定する
        obj.modifiers['subd'].levels = 1
        bpy.context.scene.collection.objects.link(obj)

def setAnimetion(prt,frame):
    for index,r in enumerate(prt):
        obj = bpy.data.objects.get(str(index))
        obj.keyframe_insert(data_path = "location",frame=frame)
        obj.location = r

def CFD_plot(x,y,z):
    vx = np.zeros((x + 3, y + 2, z + 2), dtype=np.float64)  # x速度
    vy = np.zeros((x + 2, y + 3, z + 2), dtype=np.float64)  # y速度
    vz = np.zeros((x + 2, y + 2, z + 3), dtype=np.float64)  # z速度
    p = np.zeros((x + 2, y + 2, z + 2), dtype=np.float64)  # 圧力
    s = np.zeros((x + 2, y + 2, z + 2), dtype=np.float64)  # ダイバージェンス
    prt = np.random.rand(2048*2, 3) * np.array((x,y,z),dtype=np.float64)+1  # 粒子の初期座標、x,y,z
    makeParticle(prt)
    frame = 0
    while frame < maxframe:
        #移流
        Advection(x, y, z, vx, vy, vz)
        #粘性
        Viscosity(x, y, z, vx, vy, vz)
        #外力
        vy[5,5,1]+=1
        #ダイバージェンス計算
        Div(x, y, z, vx, vy, vz, s)
        #圧力計算
        Poisson(x, y, z, p, s)
        #修正
        Rhs(x, y, z, vx, vy, vz, p)
        #可視化
        Flowparticles(vx, vy, vz, prt)
        # 逐次画面更新
        if frame%20==0:
            setAnimetion(prt,frame)
        # print
        frame += 1

if __name__ == "__main__":
    CFD_plot(10,10,10)