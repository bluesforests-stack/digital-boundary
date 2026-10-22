/* p5.js + ml5 PoseNet + Matter.js. No video pixels are rendered. */
const {Engine, Bodies, Body, Composite, Events} = Matter;
const COUNT = 220, MARGIN = 21, STEP = 1000 / 120;
let engine, particles = [], walls = [], bits = [], ripples = [];
let accumulator = 0, poseNet, stream, video, cameraState = 'off';
let targets = [], lastPoseTime = -Infinity, previous = new Map();
let pointerPrevious = null, wasActive = false;
const statusNode = document.getElementById('status');
const cameraButton = document.getElementById('camera');

function setup() {
  createCanvas(windowWidth, windowHeight); pixelDensity(Math.min(devicePixelRatio, 2));
  textFont('monospace'); textAlign(CENTER, CENTER);
  engine = Engine.create({positionIterations:6, velocityIterations:4});
  engine.gravity.y = 0;
  makeWalls();
  for (let i = 0; i < COUNT; i++) {
    const r = random(4.3, 6.4);
    const body = Bodies.circle(random(MARGIN+12,width-MARGIN-12), random(MARGIN+12,height-MARGIN-12), r, {
      restitution:0.88, friction:0.015, frictionAir:0.022, density:0.0018,
      label:'letter', slop:0.02
    });
    Composite.add(engine.world, body);
    particles.push({body, r, glyph:String.fromCharCode(65+i%26), nextChange:random(300), phase:random(TWO_PI)});
  }
  Events.on(engine, 'collisionStart', event => {
    for (const pair of event.pairs) {
      const wall = pair.bodyA.label.startsWith('wall:') ? pair.bodyA : pair.bodyB.label.startsWith('wall:') ? pair.bodyB : null;
      if (!wall) continue;
      const particle = wall === pair.bodyA ? pair.bodyB : pair.bodyA;
      const side = wall.label.split(':')[1], p = particle.position;
      const x = side==='left' ? MARGIN : side==='right' ? width-MARGIN : constrain(p.x,MARGIN,width-MARGIN);
      const y = side==='top' ? MARGIN : side==='bottom' ? height-MARGIN : constrain(p.y,MARGIN,height-MARGIN);
      ripples.push({x,y,side,born:millis(),strength:constrain(particle.speed/8,0.35,1)});
      if (ripples.length > 65) ripples.shift();
    }
  });
}

function makeWalls() {
  walls.forEach(w=>Composite.remove(engine.world,w));
  const t=100;
  walls=[
    Bodies.rectangle(MARGIN-t/2,height/2,t,height+2*t,{isStatic:true,label:'wall:left'}),
    Bodies.rectangle(width-MARGIN+t/2,height/2,t,height+2*t,{isStatic:true,label:'wall:right'}),
    Bodies.rectangle(width/2,MARGIN-t/2,width+2*t,t,{isStatic:true,label:'wall:top'}),
    Bodies.rectangle(width/2,height-MARGIN+t/2,width+2*t,t,{isStatic:true,label:'wall:bottom'})
  ];
  Composite.add(engine.world,walls); bits=[];
  for(let x=MARGIN;x<=width-MARGIN;x+=14) {
    bits.push({x,y:MARGIN,side:'top',seed:random(1000)});
    bits.push({x,y:height-MARGIN,side:'bottom',seed:random(1000)});
  }
  for(let y=MARGIN+14;y<height-MARGIN;y+=14) {
    bits.push({x:MARGIN,y,side:'left',seed:random(1000)});
    bits.push({x:width-MARGIN,y,side:'right',seed:random(1000)});
  }
}

// Webcam is opened only on an explicit button gesture; errors remain visible.
cameraButton.onclick = async () => {
  if(cameraState==='on' || cameraState==='loading') { stopCamera(); return; }
  if(!window.isSecureContext || !navigator.mediaDevices) {
    statusNode.textContent='摄像头需要 HTTPS 或 localhost · 仍可拖动鼠标体验'; return;
  }
  cameraState='loading'; cameraButton.textContent='取消摄像头';
  statusNode.textContent='正在请求摄像头并加载 PoseNet…';
  try {
    const requested = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480,facingMode:'user'},audio:false});
    if(cameraState!=='loading') { requested.getTracks().forEach(t=>t.stop()); return; }
    stream=requested;
    video=document.createElement('video'); video.muted=true; video.playsInline=true;
    video.width=640; video.height=480; video.srcObject=stream; document.body.appendChild(video);
    await video.play();
    if(cameraState!=='loading') return;
    const source=video;
    const model=ml5.poseNet(source,{detectionType:'single',architecture:'MobileNetV1',outputStride:16,inputResolution:257,multiplier:0.75,minConfidence:0.35,flipHorizontal:false},()=>{
      if(video!==source || cameraState!=='loading') return;
      cameraState='on'; cameraButton.textContent='关闭摄像头';
    });
    poseNet=model;
    model.on('pose', results=>{
      if(cameraState!=='on' || video!==source) return;
      readPose(results);
    });
  } catch(error) {
    stopCamera(); statusNode.textContent='摄像头未就绪：'+error.message+' · 可拖动鼠标体验';
  }
};
function stopCamera() {
  cameraState='off';
  if(poseNet) poseNet.removeAllListeners('pose');
  poseNet=null;
  if(stream) stream.getTracks().forEach(t=>t.stop());
  if(video) { video.pause(); video.srcObject=null; video.remove(); }
  stream=null;video=null; targets=[];previous.clear();lastPoseTime=-Infinity;
  cameraButton.textContent='开启摄像头';
  statusNode.textContent='静默状态 · 字符缓慢飘落 / 拖动鼠标可体验';
}

function readPose(results) {
  const now=millis(), pose=results[0]?.pose;
  if(!pose || pose.score<0.2) {targets=[];previous.clear();return;}
  const names=['nose','leftShoulder','rightShoulder','leftWrist','rightWrist'];
  const next=[];
  for(const key of pose.keypoints) {
    if(!names.includes(key.part) || key.score<0.4) continue;
    // Stretch to canvas and mirror exactly once, so movement feels like a mirror.
    const x=map(key.position.x,0,video.videoWidth||640,width-MARGIN-12,MARGIN+12);
    const y=map(key.position.y,0,video.videoHeight||480,MARGIN+12,height-MARGIN-12);
    const old=previous.get(key.part);
    const dt=old?Math.max(16,now-old.time):33;
    const vx=old?constrain((x-old.x)/dt*16.667,-22,22):0;
    const vy=old?constrain((y-old.y)/dt*16.667,-22,22):0;
    next.push({x,y,vx,vy});previous.set(key.part,{x,y,time:now});
  }
  const valid=new Set(pose.keypoints.filter(k=>k.score>=0.4).map(k=>k.part));
  for(const name of previous.keys()) if(!valid.has(name)) previous.delete(name);
  targets=next;if(next.length) lastPoseTime=now;
}

function getTargets() {
  if(cameraState==='on' && millis()-lastPoseTime<550 && targets.length) return targets;
  if(mouseIsPressed && mouseY<height-85) {
    const old=pointerPrevious||{x:mouseX,y:mouseY};
    const t={x:constrain(mouseX,MARGIN+12,width-MARGIN-12),y:constrain(mouseY,MARGIN+12,height-MARGIN-12),vx:constrain(mouseX-old.x,-22,22),vy:constrain(mouseY-old.y,-22,22)};
    pointerPrevious={x:mouseX,y:mouseY};return [t];
  }
  pointerPrevious=null;return [];
}

function simulate(points) {
  const active=points.length>0;
  for(let i=0;i<particles.length;i++) {
    const p=particles[i],b=p.body;
    if(active) {
      const target=points[i%points.length];
      const motion=Math.hypot(target.vx,target.vy);
      const angle=p.phase+millis()*0.0007;
      const spread=22+(i%19)*2;
      const dx=target.x+cos(angle)*spread-b.position.x;
      const dy=target.y+sin(angle)*spread-b.position.y;
      // Spring attraction + motion impulse. Scale force by mass, so acceleration stays stable.
      const gain=0.000012;
      Body.applyForce(b,b.position,{
        x:b.mass*(constrain(dx*gain,-0.004,0.004)+target.vx*0.00011+random(-1,1)*(0.00006+motion*0.000018)),
        y:b.mass*(constrain(dy*gain,-0.004,0.004)+target.vy*0.00011+random(-1,1)*(0.00006+motion*0.000018))
      });
    } else {
      Body.applyForce(b,b.position,{x:b.mass*sin(millis()*0.0005+p.phase)*0.000025,y:b.mass*0.000025});
      // Recycle only in idle mode. Active letters always remain inside the walls.
      if(b.position.y>height-MARGIN-p.r-2 && b.speed<1.4) {
        Body.setPosition(b,{x:random(MARGIN+12,width-MARGIN-12),y:MARGIN+12});
        Body.setVelocity(b,{x:random(-0.3,0.3),y:random(0.2,0.7)});
      }
    }
    // Speed cap + two 120 Hz steps at 60 fps prevents tunnelling through the wall.
    if(b.speed>18) Body.setVelocity(b,{x:b.velocity.x/b.speed*18,y:b.velocity.y/b.speed*18});
  }
  Engine.update(engine, STEP);
  for(const p of particles) {
    const b=p.body,loX=MARGIN+p.r,hiX=width-MARGIN-p.r,loY=MARGIN+p.r,hiY=height-MARGIN-p.r;
    // Numerical safety: enforce containment after tab stalls or resize.
    const x=constrain(b.position.x,loX,hiX),y=constrain(b.position.y,loY,hiY);
    if(x!==b.position.x || y!==b.position.y) {
      const vx=x!==b.position.x?-b.velocity.x*0.88:b.velocity.x;
      const vy=y!==b.position.y?-b.velocity.y*0.88:b.velocity.y;
      Body.setPosition(b,{x,y});Body.setVelocity(b,{x:vx,y:vy});
    }
  }
}

function draw() {
  background(0);
  const points=getTargets(), active=points.length>0;
  // Fixed timestep, max 6 substeps per frame: stable physics without a catch-up spiral.
  accumulator+=Math.min(deltaTime,50);
  while(accumulator>=STEP) {simulate(points);accumulator-=STEP;}
  if(cameraState==='on') statusNode.textContent=active?'已感知身体 · 移动 / 挥手 / 触碰边界':'正在寻找身体 · 站在摄像头前';
  const now=millis();
  for(const p of particles) {
    if(now>p.nextChange) {
      p.glyph=active?String.fromCharCode(65+floor(random(26))):random()<0.3?String(floor(random(2))):String.fromCharCode(65+floor(random(26)));
      p.nextChange=now+random(90,400);
    }
    fill(active?170:75,active?239:110,active?187:91,active?210:130);
    noStroke();textSize(p.r*2.15);text(p.glyph,p.body.position.x,p.body.position.y);
  }
  ripples=ripples.filter(r=>now-r.born<850);
  for(const bit of bits) {
    let glow=0;
    for(const r of ripples) {
      if(r.side!==bit.side)continue;
      const d=dist(bit.x,bit.y,r.x,r.y),age=(now-r.born)/850;
      glow=Math.max(glow,Math.exp(-Math.abs(d-age*140)/22)*(1-age)*r.strength);
    }
    const flicker=noise(bit.seed,now*0.004);
    fill(90+glow*165,120+glow*135,100+glow*155,35+flicker*100+glow*120);
    textSize(10+glow*3);text(noise(bit.seed, floor(now/110))>0.5?'1':'0',bit.x,bit.y);
  }
  noFill();strokeWeight(1);
  for(const r of ripples) {
    const age=(now-r.born)/850;
    stroke(169,255,193,(1-age)*120*r.strength);circle(r.x,r.y,6+age*145);
    stroke(169,255,193,(1-age)*45*r.strength);circle(r.x,r.y,3+age*98);
  }
  wasActive=active;
}
function windowResized() {
  const oldW=width,oldH=height;
  resizeCanvas(windowWidth,windowHeight);makeWalls();
  for(const p of particles) Body.setPosition(p.body,{
    x:constrain(p.body.position.x/oldW*width,MARGIN+p.r,width-MARGIN-p.r),
    y:constrain(p.body.position.y/oldH*height,MARGIN+p.r,height-MARGIN-p.r)
  });
  targets=[];previous.clear();ripples=[];accumulator=0;
}
window.addEventListener('pagehide',stopCamera);
