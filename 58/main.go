package main

import (
	"flag"
	"fmt"
	"gaussian_plume/plume"
	"os"
	"strconv"
)

func printUsage() {
	fmt.Println("高斯烟羽模型 - 污染物浓度预测")
	fmt.Println()
	fmt.Println("用法:")
	fmt.Println("  gaussian_plume [选项] 下风向距离X 横风向距离Y 高度Z")
	fmt.Println()
	fmt.Println("排放源参数:")
	fmt.Println("  -H float       排放源高度 (m) (默认 100)")
	fmt.Println("  -Q float       排放率 (g/s) (默认 100)")
	fmt.Println("  -L float       混合层高度 (m) (默认 1000)")
	fmt.Println("  -ref int       反射次数 (默认 10)")
	fmt.Println()
	fmt.Println("气象条件:")
	fmt.Println("  -u float       风速 (m/s) (默认 5)")
	fmt.Println("  -class string  稳定度等级 (A/B/C/D/E/F) (默认 D)")
	fmt.Println()
	fmt.Println("地形参数:")
	fmt.Println("  -terrain string  地形类型 (flat/hilly/mountainous/valley) (默认 flat)")
	fmt.Println("  -slope float     地形坡度 (度) (默认 0)")
	fmt.Println("  -elev float      地形高程 (m) (默认 0)")
	fmt.Println("  -rough float     地表粗糙度 (m) (默认 0)")
	fmt.Println("  -valleyW float   山谷宽度 (m) (默认 1000)")
	fmt.Println("  -upwind          迎风坡设置 (默认 false)")
	fmt.Println("  -amp float       自定义扩散放大因子 (默认 0=自动计算)")
	fmt.Println()
	fmt.Println("其他选项:")
	fmt.Println("  -max           计算最大地面浓度及位置")
	fmt.Println("  -h, -help      显示帮助信息")
	fmt.Println()
	fmt.Println("稳定度等级说明:")
	fmt.Println("  A - 极不稳定")
	fmt.Println("  B - 中等不稳定")
	fmt.Println("  C - 弱不稳定")
	fmt.Println("  D - 中性")
	fmt.Println("  E - 弱稳定")
	fmt.Println("  F - 中等稳定")
	fmt.Println()
	fmt.Println("示例:")
	fmt.Println("  gaussian_plume -H 80 -Q 150 -u 3 -class B 1000 0 0")
	fmt.Println("  gaussian_plume -max -H 100 -Q 100 -u 5 -class D -terrain hilly -slope 15")
	fmt.Println("  gaussian_plume -terrain mountainous -slope 25 -elev 500 -u 5 2000 0 0")
	fmt.Println("  gaussian_plume -terrain valley -valleyW 2000 -u 8 3000 0 0")
}

func main() {
	height := flag.Float64("H", 100, "排放源高度 (m)")
	emissionRate := flag.Float64("Q", 100, "排放率 (g/s)")
	mixingHeight := flag.Float64("L", 1000, "混合层高度 (m)")
	reflectionCount := flag.Int("ref", 10, "反射次数")
	windSpeed := flag.Float64("u", 5, "风速 (m/s)")
	stabilityClass := flag.String("class", "D", "稳定度等级 (A/B/C/D/E/F)")
	terrainType := flag.String("terrain", "flat", "地形类型 (flat/hilly/mountainous/valley)")
	slope := flag.Float64("slope", 0, "地形坡度 (度)")
	elevation := flag.Float64("elev", 0, "地形高程 (m)")
	roughness := flag.Float64("rough", 0, "地表粗糙度 (m)")
	valleyWidth := flag.Float64("valleyW", 1000, "山谷宽度 (m)")
	isUpwind := flag.Bool("upwind", false, "迎风坡设置")
	amplification := flag.Float64("amp", 0, "自定义扩散放大因子")
	calcMax := flag.Bool("max", false, "计算最大地面浓度及位置")
	help := flag.Bool("h", false, "显示帮助信息")
	flag.BoolVar(help, "help", false, "显示帮助信息")

	flag.Usage = printUsage
	flag.Parse()

	if *help {
		printUsage()
		return
	}

	var class plume.StabilityClass
	switch *stabilityClass {
	case "A", "a":
		class = plume.ClassA
	case "B", "b":
		class = plume.ClassB
	case "C", "c":
		class = plume.ClassC
	case "D", "d":
		class = plume.ClassD
	case "E", "e":
		class = plume.ClassE
	case "F", "f":
		class = plume.ClassF
	default:
		fmt.Printf("错误: 无效的稳定度等级 '%s'\n", *stabilityClass)
		printUsage()
		os.Exit(1)
	}

	var tType plume.TerrainType
	switch *terrainType {
	case "flat":
		tType = plume.TerrainFlat
	case "hilly":
		tType = plume.TerrainHilly
	case "mountainous":
		tType = plume.TerrainMountainous
	case "valley":
		tType = plume.TerrainValley
	default:
		fmt.Printf("错误: 无效的地形类型 '%s'\n", *terrainType)
		printUsage()
		os.Exit(1)
	}

	terrain := plume.TerrainParams{
		Type:              tType,
		Slope:             *slope,
		Elevation:         *elevation,
		Roughness:         *roughness,
		ValleyWidth:       *valleyWidth,
		IsUpwind:         *isUpwind,
		AmplificationFactor: *amplification,
	}

	source := plume.SourceParams{
		Height:         *height,
		EmissionRate:   *emissionRate,
		MixingHeight:   *mixingHeight,
		ReflectionCount: *reflectionCount,
		Terrain:        terrain,
	}

	weather := plume.WeatherParams{
		WindSpeed:      *windSpeed,
		StabilityClass: class,
	}

	fmt.Println("=== 高斯烟羽模型计算结果 ===")
	fmt.Printf("排放源高度: %.2f m\n", source.Height)
	fmt.Printf("排放率: %.2f g/s\n", source.EmissionRate)
	fmt.Printf("混合层高度: %.2f m\n", source.MixingHeight)
	fmt.Printf("反射次数: %d\n", source.ReflectionCount)
	fmt.Printf("风速: %.2f m/s\n", weather.WindSpeed)
	fmt.Printf("稳定度等级: %c\n", rune(weather.StabilityClass))
	fmt.Println()
	fmt.Println("--- 地形参数 ---")
	fmt.Printf("地形类型: %s\n", *terrainType)
	fmt.Printf("地形坡度: %.2f 度\n", terrain.Slope)
	fmt.Printf("地形高程: %.2f m\n", terrain.Elevation)
	fmt.Printf("地表粗糙度: %.2f m\n", terrain.Roughness)
	if tType == plume.TerrainValley {
		fmt.Printf("山谷宽度: %.2f m\n", terrain.ValleyWidth)
	}
	fmt.Printf("迎风坡: %t\n", terrain.IsUpwind)
	if terrain.AmplificationFactor > 0 {
		fmt.Printf("自定义放大因子: %.2f\n", terrain.AmplificationFactor)
	}
	fmt.Println()

	if *calcMax {
		maxConc, maxX := plume.MaxGroundLevelConcentration(source, weather)
		fmt.Printf("最大地面浓度: %.6f g/m³\n", maxConc)
		fmt.Printf("出现位置: 下风向 %.0f m 处\n", maxX)
		return
	}

	args := flag.Args()
	if len(args) < 3 {
		fmt.Println("错误: 请提供坐标参数 X Y Z")
		printUsage()
		os.Exit(1)
	}

	x, err := strconv.ParseFloat(args[0], 64)
	if err != nil {
		fmt.Printf("错误: 无效的X坐标 '%s'\n", args[0])
		os.Exit(1)
	}

	y, err := strconv.ParseFloat(args[1], 64)
	if err != nil {
		fmt.Printf("错误: 无效的Y坐标 '%s'\n", args[1])
		os.Exit(1)
	}

	z, err := strconv.ParseFloat(args[2], 64)
	if err != nil {
		fmt.Printf("错误: 无效的Z坐标 '%s'\n", args[2])
		os.Exit(1)
	}

	point := plume.Point{X: x, Y: y, Z: z}
	concentration := plume.CalculateConcentration(source, weather, point)

	fmt.Printf("计算点坐标: (%.2f, %.2f, %.2f) m\n", point.X, point.Y, point.Z)
	fmt.Printf("污染物浓度: %.6f g/m³\n", concentration)

	if point.Z == 0 {
		fmt.Println("(地面浓度)")
	}
}
