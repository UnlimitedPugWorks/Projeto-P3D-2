/**
 * common.glsl
 * Common types and functions used for ray tracing.
 */

const float pi = 3.14159265358979;
const float epsilon = 0.001;

struct Ray {
    vec3 o;     // origin
    vec3 d;     // direction - always set with normalized vector
    float t;    // time, for motion blur
};

Ray createRay(vec3 o, vec3 d, float t)
{
    Ray r;
    r.o = o;
    r.d = d;
    r.t = t;
    return r;
}

Ray createRay(vec3 o, vec3 d)
{
    return createRay(o, d, 0.0);
}

vec3 pointOnRay(Ray r, float t)
{
    return r.o + r.d * t;
}

float gSeed = 0.0;

uint baseHash(uvec2 p)
{
    p = 1103515245U * ((p >> 1U) ^ (p.yx));
    uint h32 = 1103515245U * ((p.x) ^ (p.y>>3U));
    return h32 ^ (h32 >> 16);
}

float hash1(inout float seed) {
    uint n = baseHash(floatBitsToUint(vec2(seed += 0.1,seed += 0.1)));
    return float(n) / float(0xffffffffU);
}

vec2 hash2(inout float seed) {
    uint n = baseHash(floatBitsToUint(vec2(seed += 0.1,seed += 0.1)));
    uvec2 rz = uvec2(n, n * 48271U);
    return vec2(rz.xy & uvec2(0x7fffffffU)) / float(0x7fffffff);
}

vec3 hash3(inout float seed)
{
    uint n = baseHash(floatBitsToUint(vec2(seed += 0.1, seed += 0.1)));
    uvec3 rz = uvec3(n, n * 16807U, n * 48271U);
    return vec3(rz & uvec3(0x7fffffffU)) / float(0x7fffffff);
}

float rand(vec2 v)
{
    return fract(sin(dot(v.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 toLinear(vec3 c)
{
    return pow(c, vec3(2.2));
}

vec3 toGamma(vec3 c)
{
    return pow(c, vec3(1.0 / 2.2));
}

vec2 randomInUnitDisk(inout float seed) {
    vec2 h = hash2(seed) * vec2(1.0, 6.28318530718);
    float phi = h.y;
    float r = sqrt(h.x);
	return r * vec2(sin(phi), cos(phi));
}

vec3 randomInUnitSphere(inout float seed)
{
    vec3 h = hash3(seed) * vec3(2.0, 6.28318530718, 1.0) - vec3(1.0, 0.0, 0.0);
    float phi = h.y;
    float r = pow(h.z, 1.0/3.0);
	return r * vec3(sqrt(1.0 - h.x * h.x) * vec2(sin(phi), cos(phi)), h.x);
}

vec3 randomUnitVector(inout float seed) //to be used in diffuse reflections with distribution cosine
{
    return(normalize(randomInUnitSphere(seed)));
}

struct Camera
{
    vec3 eye;
    vec3 u, v, n;
    float width, height;
    float lensRadius;
    float planeDist, focusDist;
    float time0, time1;
    vec3 left_corner;
    vec3 horizontal;
    vec3 vertical;
};

Camera createCamera(
    vec3 eye,
    vec3 at,
    vec3 worldUp,
    float fovy,
    float aspect,
    float aperture,  //diametro em multiplos do pixel size
    float focusDist,  //focal ratio
    float time0,
    float time1)
{
    Camera cam;
    if(aperture == 0.0) cam.focusDist = 1.0; //pinhole camera then focus in on vis plane
    else cam.focusDist = focusDist;
    vec3 w = eye - at;
    cam.planeDist = length(w);
    cam.height = 2.0 * cam.planeDist * tan(fovy * pi / 180.0 * 0.5);
    cam.width = aspect * cam.height;

    cam.lensRadius = aperture * 0.5 * cam.width / iResolution.x;  //aperture ratio * pixel size; (1 pixel=lente raio 0.5)
    cam.eye = eye;
    cam.n = normalize(w);
    cam.u = normalize(cross(worldUp, cam.n));
    cam.v = cross(cam.n, cam.u);
    cam.time0 = time0;
    cam.time1 = time1;
    cam.horizontal = focusDist * cam.width * cam.u;
    cam.vertical = focusDist * cam.height * cam.v;
    cam.left_corner = eye - cam.horizontal * 0.5  - cam.vertical * 0.5 - focusDist * w;

    return cam;
}

Ray getRay(Camera cam, vec2 pixel_sample){  //rnd pixel_sample viewport coordinates

    vec2 ls = cam.lensRadius * randomInUnitDisk(gSeed);  //ls - lens sample for DOF
    float time = cam.time0 + hash1(gSeed) * (cam.time1 - cam.time0);
    
    //Calculate eye_offset and ray direction
    vec3 eye_offset = cam.eye + cam.u * ls.x + cam.v * ls.y;


    vec3 ray_direction = vec3
                            ( cam.width * cam.u  * ((pixel_sample.x/iResolution.x) - 0.5) +
                            cam.height * cam.v * ((pixel_sample.y/iResolution.y) - 0.5) -
                            cam.n * cam.planeDist * cam.focusDist) - vec3(ls, 0.0);

    

    return createRay(eye_offset, normalize(ray_direction), time);
}

// MT_ material type
#define MT_DIFFUSE 0
#define MT_METAL 1
#define MT_DIALECTRIC 2

struct Material
{
    int type;
    vec3 albedo;  //diffuse color
    vec3 specColor;  //the color tint for specular reflections. for metals and opaque dieletrics like coloured glossy plastic
    vec3 emissive; //
    float roughness; // controls roughness for metals. It can be used for rough refractions
    float refIdx; // index of refraction for dialectric
    vec3 refractColor; // absorption for beer's law
};

Material createDiffuseMaterial(vec3 albedo){
    Material m;
    m.type = MT_DIFFUSE;
    m.albedo = albedo;
    m.specColor = vec3(0.0);
    m.roughness = 1.0;  //ser usado na iluminação direta
    m.refIdx = 1.0;
    m.refractColor = vec3(0.0);
    m.emissive = vec3(0.0);
    return m;
}

Material createMetalMaterial(vec3 specClr, float roughness){
    Material m;
    m.type = MT_METAL;
    m.albedo = vec3(0.0);
    m.specColor = specClr;
    m.roughness = roughness;
    m.emissive = vec3(0.0);
    return m;
}

Material createDialectricMaterial(vec3 refractClr, float refIdx, float roughness){
    Material m;
    m.type = MT_DIALECTRIC;
    m.albedo = vec3(0.0);
    m.specColor = vec3(0.04);
    m.refIdx = refIdx;
    m.refractColor = refractClr;  
    m.roughness = roughness;
    m.emissive = vec3(0.0);
    return m;
}

struct HitRecord{
    vec3 pos;
    vec3 normal;
    float t;            // ray parameter
    Material material;
};

float schlick(float cosine, float refIdx){
    float ni = 1.0; // assume medium is air
    float nt = refIdx;

	float r0 = pow(((ni - nt) / (ni + nt)), 2.0);

	float kr = r0 + (1.0 - r0) * pow((1.0 - cosine), 5.0);
    
    return kr;
}

bool boolean_refract(const in vec3 v, const in vec3 n, const in float ni_over_nt, 
                      out vec3 refracted) {
    vec3 uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0 - ni_over_nt*ni_over_nt*(1.0 -dt*dt);
    if (discriminant > 0.0) {
        refracted = ni_over_nt*(uv - n*dt) - n * sqrt(discriminant);
        return true;
    }
    else { 
        return false;
    }
}



bool scatter(Ray rIn, HitRecord rec, out vec3 atten, out Ray rScattered){
    if(rec.material.type == MT_DIFFUSE){
        // Random diffuse scattered ray direction
        vec3 pointS = rec.pos + rec.normal + normalize(randomInUnitSphere(gSeed));
        rScattered = createRay(rec.pos, pointS - rec.pos, rIn.t);
        atten = rec.material.albedo * max(dot(rScattered.d, rec.normal), 0.0) / pi;
        return true;
    }
    else if(rec.material.type == MT_METAL){
        
        vec3 reflected = reflect(normalize(rIn.d), rec.normal);

        bool isScattered = dot(reflected, rec.normal) > 0.0;

        // considering fuzzy reflections
        rScattered = createRay(rec.pos, normalize(reflected + rec.material.roughness * randomInUnitSphere(gSeed)), rIn.t);

        atten = rec.material.specColor;

        return isScattered;
    }
    else if(rec.material.type == MT_DIALECTRIC){
        vec3 outwardNormal;
        vec3 reflected = reflect(normalize(rIn.d), rec.normal);
        float niOverNt;
        atten = vec3(1.0);
        float dotAux = dot(rIn.d, rec.normal);
        vec3 refracted;
        float cosine;
        if( dotAux > 0.0){ //hit inside
            outwardNormal = -rec.normal;
            niOverNt = rec.material.refIdx; 
            cosine = rec.material.refIdx * dotAux / length(rIn.d);
            atten = exp(-rec.material.refractColor * rec.t);   // atten = apply Beer's law by using rec.material.refractColor

        }
        else  //hit from outside
        {
            outwardNormal = rec.normal;
            niOverNt = 1.0 / rec.material.refIdx; 
            cosine = -dotAux / length(rIn.d);
        }

        //Use probabilistic math to decide if scatter a reflected ray or a refracted ray
        float reflectProb;
        
        // no total reflection
        if(boolean_refract(normalize(rIn.d), outwardNormal, niOverNt, refracted)){
            reflectProb = schlick(cosine, rec.material.refIdx);
        }
        else{
            reflectProb = 1.0;
        }

        if( hash1(gSeed) < reflectProb){  
            //Reflection
            rScattered = createRay(rec.pos, normalize(reflected), rIn.t);
        }
        else{
            //Refraction
            rScattered = createRay(rec.pos, normalize(refracted), rIn.t);
        }
        return true;
    }
    return false;
}

struct Triangle {vec3 a; vec3 b; vec3 c; };

Triangle createTriangle(vec3 v0, vec3 v1, vec3 v2)
{
    Triangle t;
    t.a = v0; t.b = v1; t.c = v2;
    return t;
}

bool hit_triangle(Triangle tri, Ray r, float tmin, float tmax, out HitRecord rec)
{
    vec3 p0 = tri.a;
    vec3 p1 = tri.b;
    vec3 p2 = tri.c;

    float a = p1.x - p0.x;
    float b = p2.x - p0.x;
    float c = -r.d.x;
    float d = r.o.x - p0.x;

    float e = p1.y - p0.y;
    float f = p2.y - p0.y;
    float g = -r.d.y;
    float h  = r.o.y - p0.y;

    float i = p1.z - p0.z;
	float j = p2.z - p0.z;
	float k = -r.d.z;
	float l = r.o.z - p0.z;

    float denom = (a * (f * k - g * j) + b * (g * i - e * k) + c * (e * j - f * i));

	float beta = (d * (f * k - g * j) + b * (g * l - h * k) + c * (h * j - f * l)) / denom;

	if (beta < 0.0) {
		return false;
	}

	float gamma = (a * (h * k - g * l) + d * (g * i - e * k) + c * (e * l - h * i)) / denom;

	if (gamma < 0.0f) {
		return false;
	}

	if (beta + gamma > 1.0f) {
		return false;
	}

	float t = (a * (f * l - h * j) + b * (h * i - e * l) +  d * (e * j - f * i )) / denom;

    if(t < tmax && t > tmin)
    {
        rec.t = t;
        rec.normal = normalize(cross((tri.b - tri.a), (tri.c - tri.a)));
        rec.pos = pointOnRay(r, rec.t);
        return true;
    }
    return false;
}


struct Sphere
{
    vec3 center;
    float radius;
};

Sphere createSphere(vec3 center, float radius)
{
    Sphere s;
    s.center = center;
    s.radius = radius;
    return s;
}


struct MovingSphere
{
    vec3 center0, center1;
    float radius;
    float time0, time1;
};

MovingSphere createMovingSphere(vec3 center0, vec3 center1, float radius, float time0, float time1)
{
    MovingSphere s;
    s.center0 = center0;
    s.center1 = center1;
    s.radius = radius;
    s.time0 = time0;
    s.time1 = time1;
    return s;
}

vec3 center(MovingSphere mvsphere, float time)
{
    return mvsphere.center0 + ((time - mvsphere.time0) / (mvsphere.time1 - mvsphere.time0)) * (mvsphere.center1 - mvsphere.center0);
}


/*
 * The function naming convention changes with these functions to show that they implement a sort of interface for
 * the book's notion of "hittable". E.g. hit_<type>.
 */

bool hit_sphere(Sphere s, Ray r, float tmin, float tmax, out HitRecord rec)
{
   vec3 dir = r.d;
   vec3 origin = r.o;
   vec3 center = s.center;
   vec3 OC = origin - center;

   float a = dot(dir,dir);
   float b = dot(dir,OC);
   float c = dot(OC, OC) - (s.radius * s.radius);
   float t;
   float discr = b * b - a *  c;

	if (discr <= 0.0f) {
		return false;
	}

    discr = sqrt(discr);
    t = (-b - discr) / a;
    if(t < tmax && t > tmin) {
        rec.t = t;
        rec.pos = pointOnRay(r, rec.t);
        rec.normal = normalize((rec.pos - center) / s.radius);
        return true;
    }
	t = (-b + discr) / a;
    if(t < tmax && t > tmin) {
        rec.t = t;
        rec.pos = pointOnRay(r, rec.t);
        rec.normal = normalize((rec.pos - center) / s.radius);
        return true;
    }

    return false;
}

bool hit_movingSphere(MovingSphere s, Ray r, float tmin, float tmax, out HitRecord rec)
{
    float a, b, c, delta;
    bool outside;
    float t;

    vec3 moving_center = center(s, r.t);

    vec3 dir = r.d;
    vec3 origin = r.o;
    vec3 OC = moving_center - origin;

    a = dot(dir, dir);
    b = dot(dir,OC);
    c = dot(OC, OC) - (s.radius * s.radius);

	float discr = b * b - a * c;

	if (discr <= 0.0f) {
		return false;
	}

    discr = sqrt(discr);
    t = (b - discr) / a;
    if(t < tmin || tmax < t){
        t = (b + discr) / a;
        if(t < tmin || tmax < t){
            return false;
        }
    }

    rec.t = t;
    rec.pos = pointOnRay(r, rec.t);
    rec.normal = normalize((rec.pos - moving_center) / s.radius);
    return true;
}


struct Box {
    vec3 vmin;
    vec3 vmax;
};

Box createBox(vec3 min, vec3 max){
    Box b;
    b.vmin = min;
    b.vmax = max;
    return b;
}

bool hit_box(Box box, Ray r, float tmin, float tmax, out HitRecord rec)
{
    // ray origin
	float ox = r.o.x;
	float oy = r.o.y;
	float oz = r.o.z;

	// ray direction
	float dx = r.d.x;
	float dy = r.d.y;
	float dz = r.d.z;

	float tx_min, ty_min, tz_min;
	float tx_max, ty_max, tz_max;

	float a = 1.0 / dx;
	float b = 1.0 / dy;
	float c = 1.0 / dz;

	if (a >= 0.0) {
		tx_min = (box.vmin.x - ox) * a;
		tx_max = (box.vmax.x - ox) * a;
	} else {
		tx_min = (box.vmax.x - ox) * a;
		tx_max = (box.vmin.x - ox) * a;
	}


	if (b >= 0.0) {
		ty_min = (box.vmin.y - oy) * b;
		ty_max = (box.vmax.y - oy) * b;
	} else {
		ty_min = (box.vmax.y - oy) * b;
		ty_max = (box.vmin.y - oy) * b;
	}

	if (c >= 0.0) {
		tz_min = (box.vmin.z - oz) * c;
		tz_max = (box.vmax.z - oz) * c;
	} else {
		tz_min = (box.vmax.z - oz) * c;
		tz_max = (box.vmin.z - oz) * c;
	}

	float tE, tL;				// Entering and leaving t values
	vec3 face_in, face_out;	    // Normals 

	// find largest tE, entering t value
	if (tx_min > ty_min) {
		tE = tx_min;

		if (a >= 0.0) {
			face_in = vec3(-1.0, 0.0, 0.0);
		} else {
			face_in = vec3(1.0, 0.0, 0.0);
		}
	}
	else {
		tE = ty_min;

		if (b >= 0.0) {
			face_in = vec3(0.0, -1.0, 0.0);
		}
		else {
			face_in = vec3(0.0, 1.0, 0.0);
		}
	}
	if (tz_min > tE) {
		tE = tz_min;

		if (c >= 0.0) {
			face_in = vec3(0.0, 0.0, -1.0);
		}
		else {
			face_in = vec3(0.0, 0.0, 1.0);
		}
	}

	// find smallest tL, leving t value
	if (tx_max < ty_max) {
		tL = tx_max;

		if (a >= 0.0) {
			face_out = vec3(1.0, 0.0, 0.0);
		} else {
			face_out = vec3(-1.0, 0.0, 0.0);
		}
	}
	else {
		tL = ty_max;
		
		if (b >= 0.0) {
			face_out = vec3(0.0, 1.0, 0.0);
		}
		else {
			face_out = vec3(0.0, -1.0, 0.0);
		}
	}

	if (tz_max < tL) {
		tL = tz_max;

		if (c >= 0.0) {
			face_out = vec3(0.0, 0.0, 1.0);
		}
		else {
			face_out = vec3(0.0, 0.0, -1.0);
		}
	}

	// condition for a hit
	if (tE < tL && tL > 0.0) {
        float t;
		if (tE > 0.0) {
            t = tE;
            if(t < tmax && t > tmin) {
                rec.t = t;
                rec.pos = pointOnRay(r, rec.t);
                rec.normal = face_in;
                return true;
            }
		}
		else {
            t = tL;
            if(t < tmax && t > tmin) {
                rec.t = t;
                rec.pos = pointOnRay(r, rec.t);
                rec.normal = face_out;
                return true;
            }
		}
	}
	else {
		return false;
	}
}

struct pointLight {
    vec3 pos;
    vec3 color;
};

pointLight createPointLight(vec3 pos, vec3 color) {
    pointLight l;
    l.pos = pos;
    l.color = color;
    return l;
}