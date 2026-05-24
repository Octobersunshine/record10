package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

const (
	G          = 6.67430e-11
	AU         = 1.496e11
	Day        = 86400
	Year       = 365.25 * Day
	SolarMass  = 1.989e30
	EarthMass  = 5.972e24
	Parsec     = 3.086e16
	Kiloparsec = 1000 * Parsec
)

type MilkyWayPotential struct {
	BulgeMass   float64
	BulgeScale  float64
	DiskMass    float64
	DiskScaleA  float64
	DiskScaleB  float64
	HaloMass    float64
	HaloScale   float64
}

func NewMilkyWayPotential() *MilkyWayPotential {
	return &MilkyWayPotential{
		BulgeMass:  1.5e10 * SolarMass,
		BulgeScale: 0.7 * Kiloparsec,
		DiskMass:   5.0e10 * SolarMass,
		DiskScaleA: 4.0 * Kiloparsec,
		DiskScaleB: 0.3 * Kiloparsec,
		HaloMass:   1.0e12 * SolarMass,
		HaloScale:  20.0 * Kiloparsec,
	}
}

func (mw *MilkyWayPotential) BulgeAcceleration(x, y, z float64) (ax, ay, az float64) {
	r := math.Sqrt(x*x + y*y + z*z)
	bs := mw.BulgeScale
	term := r * r / (bs * bs)
	factor := -G * mw.BulgeMass / (r * r * math.Pow(1+term, 1.5))
	return factor * x / r, factor * y / r, factor * z / r
}

func (mw *MilkyWayPotential) DiskAcceleration(x, y, z float64) (ax, ay, az float64) {
	R := math.Sqrt(x*x + y*y)
	a := mw.DiskScaleA
	b := mw.DiskScaleB
	term := math.Sqrt(z*z + b*b)
	denom := math.Pow(R*R+(a+term)*(a+term), 1.5)

	factorR := -G * mw.DiskMass * R / denom
	factorZ := -G * mw.DiskMass * (a+term) * z / (term * denom)

	return factorR * x / R, factorR * y / R, factorZ
}

func (mw *MilkyWayPotential) HaloAcceleration(x, y, z float64) (ax, ay, az float64) {
	r := math.Sqrt(x*x + y*y + z*z)
	rs := mw.HaloScale

	logTerm := math.Log(1 + r/rs)
	term1 := G * mw.HaloMass / (r * r)
	term2 := logTerm - (r/rs)/(1+r/rs)
	factor := -term1 * term2

	return factor * x / r, factor * y / r, factor * z / r
}

func (mw *MilkyWayPotential) Acceleration(x, y, z float64) (ax, ay, az float64) {
	ax_b, ay_b, az_b := mw.BulgeAcceleration(x, y, z)
	ax_d, ay_d, az_d := mw.DiskAcceleration(x, y, z)
	ax_h, ay_h, az_h := mw.HaloAcceleration(x, y, z)

	return ax_b + ax_d + ax_h, ay_b + ay_d + ay_h, az_b + az_d + az_h
}

func (mw *MilkyWayPotential) CircularVelocity(r float64) float64 {
	ax, _, _ := mw.Acceleration(r, 0, 0)
	return math.Sqrt(-ax * r)
}

type Body struct {
	Name string
	Mass float64
	X, Y, Z    float64
	Vx, Vy, Vz float64
	Ax, Ay, Az float64
}

func NewBody(name string, mass, x, y, z, vx, vy, vz float64) *Body {
	return &Body{
		Name: name,
		Mass: mass,
		X: x, Y: y, Z: z,
		Vx: vx, Vy: vy, Vz: vz,
	}
}

func (b *Body) Distance(other *Body) float64 {
	dx := other.X - b.X
	dy := other.Y - b.Y
	dz := other.Z - b.Z
	return math.Sqrt(dx*dx + dy*dy + dz*dz)
}

func CalculateAccelerations(bodies []*Body, mw *MilkyWayPotential) (minDist float64) {
	n := len(bodies)
	for i := range bodies {
		bodies[i].Ax, bodies[i].Ay, bodies[i].Az = 0, 0, 0

		if mw != nil {
			ax, ay, az := mw.Acceleration(bodies[i].X, bodies[i].Y, bodies[i].Z)
			bodies[i].Ax += ax
			bodies[i].Ay += ay
			bodies[i].Az += az
		}
	}

	minDist = math.Inf(1)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			a := bodies[i]
			b := bodies[j]

			dx := b.X - a.X
			dy := b.Y - a.Y
			dz := b.Z - a.Z
			distSq := dx*dx + dy*dy + dz*dz
			dist := math.Sqrt(distSq)

			if dist < minDist {
				minDist = dist
			}

			softening := 1e16
			softenedDistSq := distSq + softening*softening
			softenedDist := math.Sqrt(softenedDistSq)

			f := G * a.Mass * b.Mass / softenedDistSq
			fx := f * dx / softenedDist
			fy := f * dy / softenedDist
			fz := f * dz / softenedDist

			a.Ax += fx / a.Mass
			a.Ay += fy / a.Mass
			a.Az += fz / a.Mass
			b.Ax -= fx / b.Mass
			b.Ay -= fy / b.Mass
			b.Az -= fz / b.Mass
		}
	}
	return minDist
}

func CalculateTimestep(bodies []*Body, minDist float64, eta float64, maxDt float64) float64 {
	maxAccel := 0.0
	for _, b := range bodies {
		accelMag := math.Sqrt(b.Ax*b.Ax + b.Ay*b.Ay + b.Az*b.Az)
		if accelMag > maxAccel {
			maxAccel = accelMag
		}
	}

	if maxAccel == 0 {
		return maxDt
	}

	dt := eta * math.Sqrt(minDist/maxAccel)
	if dt > maxDt {
		dt = maxDt
	}
	minDt := 1.0
	if dt < minDt {
		dt = minDt
	}
	return dt
}

func LeapfrogStep(bodies []*Body, dt float64, mw *MilkyWayPotential) (newMinDist float64) {
	for _, b := range bodies {
		b.Vx += 0.5 * dt * b.Ax
		b.Vy += 0.5 * dt * b.Ay
		b.Vz += 0.5 * dt * b.Az
	}

	for _, b := range bodies {
		b.X += dt * b.Vx
		b.Y += dt * b.Vy
		b.Z += dt * b.Vz
	}

	newMinDist = CalculateAccelerations(bodies, mw)

	for _, b := range bodies {
		b.Vx += 0.5 * dt * b.Ax
		b.Vy += 0.5 * dt * b.Ay
		b.Vz += 0.5 * dt * b.Az
	}
	return newMinDist
}

func TotalEnergy(bodies []*Body) float64 {
	kinetic := 0.0
	potential := 0.0

	for _, b := range bodies {
		vSq := b.Vx*b.Vx + b.Vy*b.Vy + b.Vz*b.Vz
		kinetic += 0.5 * b.Mass * vSq
	}

	n := len(bodies)
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			dist := bodies[i].Distance(bodies[j])
			potential -= G * bodies[i].Mass * bodies[j].Mass / dist
		}
	}

	return kinetic + potential
}

func SolarSystem() []*Body {
	return []*Body{
		NewBody("Sun", SolarMass, 0, 0, 0, 0, 0, 0),
		NewBody("Mercury", 0.055*EarthMass,
			0.387*AU, 0, 0,
			0, 47.4e3, 0),
		NewBody("Venus", 0.815*EarthMass,
			0.723*AU, 0, 0,
			0, 35.0e3, 0),
		NewBody("Earth", EarthMass,
			AU, 0, 0,
			0, 29.78e3, 0),
		NewBody("Mars", 0.107*EarthMass,
			1.524*AU, 0, 0,
			0, 24.1e3, 0),
		NewBody("Jupiter", 317.8*EarthMass,
			5.203*AU, 0, 0,
			0, 13.1e3, 0),
		NewBody("Saturn", 95.2*EarthMass,
			9.537*AU, 0, 0,
			0, 9.7e3, 0),
	}
}

func PrintState(bodies []*Body, step int, timeDays float64) {
	fmt.Printf("\n=== Step %d | Time: %.2f days | Energy: %.6e J ===\n",
		step, timeDays, TotalEnergy(bodies))
	fmt.Printf("%-10s %15s %15s %15s %15s\n",
		"Body", "X (AU)", "Y (AU)", "Dist (AU)", "Speed (km/s)")
	fmt.Println(strings.Repeat("-", 75))

	for _, b := range bodies {
		dist := math.Sqrt(b.X*b.X + b.Y*b.Y + b.Z*b.Z) / AU
		speed := math.Sqrt(b.Vx*b.Vx + b.Vy*b.Vy + b.Vz*b.Vz) / 1000
		fmt.Printf("%-10s %15.6f %15.6f %15.6f %15.3f\n",
			b.Name, b.X/AU, b.Y/AU, dist, speed)
	}
}

func ClearScreen() {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "windows":
		cmd = exec.Command("cmd", "/c", "cls")
	default:
		cmd = exec.Command("clear")
	}
	cmd.Stdout = os.Stdout
	cmd.Run()
}

func PrintASCII(bodies []*Body, width, height int, scaleFactor float64, centerOnCluster bool) {
	canvas := make([][]byte, height)
	for i := range canvas {
		canvas[i] = make([]byte, width)
		for j := range canvas[i] {
			canvas[i][j] = ' '
		}
	}

	centerX := width / 2
	centerY := height / 2
	offsetX, offsetY := 0.0, 0.0

	if centerOnCluster && len(bodies) > 0 {
		cx, cy, _, _, _, _, _ := ClusterCenter(bodies)
		offsetX = cx
		offsetY = cy
	}

	for _, b := range bodies {
		x := int((b.X-offsetX)/Kiloparsec*scaleFactor) + centerX
		y := int(-(b.Y-offsetY)/Kiloparsec*scaleFactor) + centerY

		if x >= 0 && x < width && y >= 0 && y < height {
			if canvas[y][x] == ' ' {
				canvas[y][x] = '.'
			} else if canvas[y][x] == '.' {
				canvas[y][x] = 'o'
			} else if canvas[y][x] == 'o' {
				canvas[y][x] = 'O'
			}
		}
	}

	if !centerOnCluster {
		gcX := centerX
		gcY := centerY
		if gcX >= 0 && gcX < width && gcY >= 0 && gcY < height {
			canvas[gcY][gcX] = '+'
		}
	}

	fmt.Println()
	for _, row := range canvas {
		fmt.Println(string(row))
	}
}

func PrintClusterState(bodies []*Body, mw *MilkyWayPotential, step int, timeMyr float64) {
	cx, cy, cz, vx, vy, vz, totalMass := ClusterCenter(bodies)
	galactocentricR := math.Sqrt(cx*cx + cy*cy)
	galactocentricV := math.Sqrt(vx*vx + vy*vy)
	boundCount, tidalRadius := CountBoundStars(bodies, mw)

	fmt.Printf("\n=== Step %d | Time: %.2f Myr ===\n", step, timeMyr)
	fmt.Printf("Cluster Center: (%.2f, %.2f, %.2f) kpc\n",
		cx/Kiloparsec, cy/Kiloparsec, cz/Kiloparsec)
	fmt.Printf("Galactocentric Radius: %.2f kpc | Speed: %.1f km/s\n",
		galactocentricR/Kiloparsec, galactocentricV/1000)
	fmt.Printf("Bound Stars: %d / %d | Tidal Radius: %.2f pc\n",
		boundCount, len(bodies), tidalRadius/Parsec)
	fmt.Printf("Cluster Mass: %.2e M_sun\n", totalMass/SolarMass)
}

func BinarySystemTest() []*Body {
	m1 := 1.0 * SolarMass
	m2 := 1.0 * EarthMass
	sep := 1.0e8
	vOrb := math.Sqrt(G * m1 / sep)

	return []*Body{
		NewBody("Star1", m1, -sep/2, 0, 0, 0, vOrb*m2/(m1+m2), 0),
		NewBody("Star2", m2, sep/2, 0, 0, 0, -vOrb*m1/(m1+m2), 0),
	}
}

func CloseEncounterTest() []*Body {
	m1 := 1.0 * SolarMass
	m2 := 0.1 * SolarMass
	sep := 5.0e9

	vx := 100e3

	return []*Body{
		NewBody("Primary", m1, 0, 0, 0, 0, 0, 0),
		NewBody("Secondary", m2, -sep, sep/5, 0, vx, 0, 0),
	}
}

func HyperbolicEncounterTest() []*Body {
	m1 := 10.0 * EarthMass
	m2 := 1.0 * EarthMass
	impactParam := 1.0e7
	vInf := 10e3

	return []*Body{
		NewBody("A", m1, 0, 0, 0, vInf/2, 0, 0),
		NewBody("B", m2, -1.0e8, impactParam, 0, -vInf/2, 0, 0),
	}
}

func PlummerRadius(scaleRadius float64, u float64) float64 {
	return scaleRadius / math.Sqrt(math.Pow(u, -2.0/3.0)-1.0)
}

func PlummerCluster(N int, totalMass float64, scaleRadius float64,
	centerX, centerY, centerZ, vx, vy, vz float64, seed int64) []*Body {

	bodies := make([]*Body, N)
	rng := rand.New(rand.NewSource(seed))

	massPerParticle := totalMass / float64(N)

	for i := 0; i < N; i++ {
		u := rng.Float64()
		r := PlummerRadius(scaleRadius, u)

		theta := math.Acos(2*rng.Float64() - 1)
		phi := 2 * math.Pi * rng.Float64()

		x := r * math.Sin(theta) * math.Cos(phi)
		y := r * math.Sin(theta) * math.Sin(phi)
		z := r * math.Cos(theta)

		vEscape := math.Sqrt(2 * G * totalMass /
			math.Sqrt(r*r + scaleRadius*scaleRadius))

		var v, g float64
		for {
			v = 0.1 * rng.Float64()
			g = rng.Float64()
			if g < v*v*math.Pow(1-v*v, 3.5) {
				break
			}
		}
		vMag := v * vEscape

		vTheta := math.Acos(2*rng.Float64() - 1)
		vPhi := 2 * math.Pi * rng.Float64()

		vxLocal := vMag * math.Sin(vTheta) * math.Cos(vPhi)
		vyLocal := vMag * math.Sin(vTheta) * math.Sin(vPhi)
		vzLocal := vMag * math.Cos(vTheta)

		bodies[i] = NewBody(
			fmt.Sprintf("Star%d", i),
			massPerParticle,
			centerX+x, centerY+y, centerZ+z,
			vx+vxLocal, vy+vyLocal, vz+vzLocal,
		)
	}

	return bodies
}

func TidalRadius(clusterMass, galactocentricDist float64,
	mw *MilkyWayPotential) float64 {

	ax, _, _ := mw.Acceleration(galactocentricDist, 0, 0)
	omega2 := -ax / galactocentricDist

	dr := 0.01 * galactocentricDist
	axOuter, _, _ := mw.Acceleration(galactocentricDist+dr, 0, 0)
	dOmega2dr := (-axOuter/(galactocentricDist+dr) - omega2) / dr

	dlnOmega2dlnR := galactocentricDist / omega2 * dOmega2dr
	gamma := 1 - (1/3.0)*dlnOmega2dlnR

	rt := math.Pow(G*clusterMass/(gamma*omega2), 1.0/3.0)
	return rt
}

func TidalForce(x, y, z float64, mw *MilkyWayPotential) (tx, ty, tz float64) {
	dr := 1.0 * Parsec

	ax0, ay0, az0 := mw.Acceleration(x, y, z)

	axX, _, _ := mw.Acceleration(x+dr, y, z)
	_, ayY, _ := mw.Acceleration(x, y+dr, z)
	_, _, azZ := mw.Acceleration(x, y, z+dr)

	daxdx := (axX - ax0) / dr
	daydy := (ayY - ay0) / dr
	dazdz := (azZ - az0) / dr

	r := math.Sqrt(x*x + y*y + z*z)
	if r > 0 {
		tx = -daxdx * x
		ty = -daydy * y
		tz = -dazdz * z
	}
	return tx, ty, tz
}

func ClusterCenter(bodies []*Body) (cx, cy, cz, vx, vy, vz, totalMass float64) {
	totalMass = 0.0
	cx, cy, cz = 0, 0, 0
	vx, vy, vz = 0, 0, 0

	for _, b := range bodies {
		totalMass += b.Mass
		cx += b.X * b.Mass
		cy += b.Y * b.Mass
		cz += b.Z * b.Mass
		vx += b.Vx * b.Mass
		vy += b.Vy * b.Mass
		vz += b.Vz * b.Mass
	}

	if totalMass > 0 {
		cx /= totalMass
		cy /= totalMass
		cz /= totalMass
		vx /= totalMass
		vy /= totalMass
		vz /= totalMass
	}
	return cx, cy, cz, vx, vy, vz, totalMass
}

func CountBoundStars(bodies []*Body, mw *MilkyWayPotential) (boundCount int, tidalRadius float64) {
	cx, cy, _, _, _, _, totalMass := ClusterCenter(bodies)
	galactocentricDist := math.Sqrt(cx*cx + cy*cy)
	tidalRadius = TidalRadius(totalMass, galactocentricDist, mw)

	boundCount = 0
	for _, b := range bodies {
		dx := b.X - cx
		dy := b.Y - cy
		dz := b.Z - cz
		r := math.Sqrt(dx*dx + dy*dy + dz*dz)
		if r < tidalRadius {
			boundCount++
		}
	}
	return boundCount, tidalRadius
}

func DwarfGalaxyOrbit(r0, v0 float64) (x, y, vx, vy float64) {
	x = r0
	y = 0
	vx = 0
	vy = v0
	return x, y, vx, vy
}

func main() {
	mw := NewMilkyWayPotential()

	N := 200
	clusterMass := 1.0e5 * SolarMass
	scaleRadius := 5.0 * Parsec

	galactocentricR := 50.0 * Kiloparsec
	vCirc := mw.CircularVelocity(galactocentricR)
	orbitSpeed := vCirc * 0.8

	cx, cy, cvx, cvy := DwarfGalaxyOrbit(galactocentricR, orbitSpeed)
	bodies := PlummerCluster(N, clusterMass, scaleRadius, cx, cy, 0, cvx, cvy, 0, 42)

	eta := 0.02
	maxDt := 0.5 * 1.0e6 * Year
	totalTime := 5.0 * 1.0e9 * Year
	outputInterval := 20.0 * 1.0e6 * Year

	Myr := 1.0e6 * Year

	minDist := CalculateAccelerations(bodies, mw)
	dt := CalculateTimestep(bodies, minDist, eta, maxDt)

	currentTime := 0.0
	nextOutput := 0.0
	stepCount := 0
	totalSteps := 0
	outputCount := 0

	fmt.Println("=== Star Cluster Tidal Stripping Simulation ===")
	fmt.Printf("Cluster: %d stars, Mass: %.1e M_sun\n", N, clusterMass/SolarMass)
	fmt.Printf("Scale Radius: %.1f pc\n", scaleRadius/Parsec)
	fmt.Printf("Galactocentric Radius: %.1f kpc\n", galactocentricR/Kiloparsec)
	fmt.Printf("Orbital Speed: %.1f km/s\n", orbitSpeed/1000)
	fmt.Printf("Total Simulation: %.1f Gyr\n", totalTime/(1e9*Year))
	fmt.Println("\nMilky Way Model:")
	fmt.Printf("  Bulge:  %.1e M_sun, Scale: %.1f kpc\n", mw.BulgeMass/SolarMass, mw.BulgeScale/Kiloparsec)
	fmt.Printf("  Disk:   %.1e M_sun, Scale: %.1f kpc\n", mw.DiskMass/SolarMass, mw.DiskScaleA/Kiloparsec)
	fmt.Printf("  Halo:   %.1e M_sun, Scale: %.1f kpc\n", mw.HaloMass/SolarMass, mw.HaloScale/Kiloparsec)
	time.Sleep(3 * time.Second)

	startTime := time.Now()

	for currentTime < totalTime {
		if currentTime >= nextOutput {
			ClearScreen()
			PrintClusterState(bodies, mw, outputCount, currentTime/Myr)
			PrintASCII(bodies, 80, 25, 0.5, true)

			fmt.Printf("\n=== Simulation Info ===\n")
			fmt.Printf("Current Timestep: %.2f Myr\n", dt/Myr)
			fmt.Printf("Total Steps: %d\n", totalSteps)
			fmt.Printf("Elapsed Real Time: %v\n", time.Since(startTime))

			nextOutput += outputInterval
			outputCount++
			time.Sleep(50 * time.Millisecond)
		}

		if currentTime+dt > totalTime {
			dt = totalTime - currentTime
		}

		minDist = LeapfrogStep(bodies, dt, mw)
		currentTime += dt
		totalSteps++

		stepCount++
		if stepCount >= 5 {
			dt = CalculateTimestep(bodies, minDist, eta, maxDt)
			stepCount = 0
		}
	}

	elapsed := time.Since(startTime)
	ClearScreen()

	fmt.Printf("\n=== Simulation Complete ===\n")
	fmt.Printf("Real Time: %v\n", elapsed)
	fmt.Printf("Total Steps: %d\n", totalSteps)
	fmt.Printf("Performance: %.2f steps/sec\n",
		float64(totalSteps)/elapsed.Seconds())

	fmt.Printf("\n=== Final State ===\n")
	PrintClusterState(bodies, mw, outputCount, currentTime/Myr)

	_, tidalRadius := CountBoundStars(bodies, mw)
	initialRt := TidalRadius(clusterMass, galactocentricR, mw)
	fmt.Printf("\n=== Tidal Evolution ===\n")
	fmt.Printf("Initial Tidal Radius: %.1f pc\n", initialRt/Parsec)
	fmt.Printf("Final Tidal Radius:   %.1f pc\n", tidalRadius/Parsec)
}
