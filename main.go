//go:build linux && amd64
// +build linux,amd64

package main

/*
	void maxmul(float *A, float* B, float *C, int size);
	#cgo LDFLAGS: -L. -L./ -lmaxmul
*/
import "C"

import (
	"net/http"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/opts"
	"github.com/go-echarts/go-echarts/v2/types"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
)

var (
	size               int
	list_size          []int
	cpu_time, gpu_time []uint64
)

func main() {
	max, _ := strconv.Atoi(os.Args[1])
	for size = 5; size < max; size += 10 {
		list_size = append(list_size, size)
		calcTime()
	}
	http.HandleFunc("/", httpserver)
	http.ListenAndServe(":6006", nil)
}

func Maxmul(a []C.float, b []C.float, c []C.float, size int) {
	C.maxmul(&a[0], &b[0], &c[0], C.int(size))
}

func genMatrix(size int) mat.Matrix {
	data := make([]float64, size*size)
	for i := range data {
		data[i] = rand.NormFloat64()
	}
	matrix := mat.NewDense(size, size, data)
	// 对矩阵 a 进行随机赋值
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			matrix.Set(i, j, rand.Float64())
		}
	}
	return matrix
}

func CPU(cpu_matrix1, cpu_matrix2 mat.Matrix) {
	var cpu_matrix3 mat.Dense
	t_start := time.Now()
	cpu_matrix3.Mul(cpu_matrix1, cpu_matrix2)
	t_end := time.Now()
	cpu_time = append(cpu_time, uint64(t_end.Sub(t_start)))
}

func GPU(gpu_matrix1, gpu_matrix2 []C.float, size int) {
	gpu_matrix3 := make([]C.float, size*size)
	t_start := time.Now()
	Maxmul(gpu_matrix1, gpu_matrix2, gpu_matrix3, size)
	t_end := time.Now()
	gpu_time = append(gpu_time, uint64(t_end.Sub(t_start)))
}

func calcTime() {
	println("size:", size)
	//生成随机矩阵
	cpu_matrix1 := genMatrix(size)
	cpu_matrix2 := genMatrix(size)
	// CPU
	runtime.GOMAXPROCS(runtime.NumCPU())
	go func() {
		CPU(cpu_matrix1, cpu_matrix2)
		println("CPU :", cpu_time[len(cpu_time)-1])
	}()

	//转换成 GPU 矩阵格式
	gpu_matrix1 := make([]C.float, size*size)
	gpu_matrix2 := make([]C.float, size*size)
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			gpu_matrix1[i*j+j] = C.float(cpu_matrix1.At(i, j))
			gpu_matrix2[i*j+j] = C.float(cpu_matrix2.At(i, j))
		}
	}
	//GPU
	GPU(gpu_matrix1, gpu_matrix2, size)
	println("GPU ：", gpu_time[len(gpu_time)-1])
}

func getLineItems(spend_time []uint64) []opts.LineData {
	items := make([]opts.LineData, 0)
	for k, _ := range spend_time {
		items = append(items, opts.LineData{Value: spend_time[k]})
	}
	return items
}

func httpserver(w http.ResponseWriter, _ *http.Request) {
	// create a new line instance
	line := charts.NewLine()
	// set some global options like Title/Legend/ToolTip or anything else
	line.SetGlobalOptions(
		charts.WithInitializationOpts(opts.Initialization{Theme: types.ThemeWesteros}),
		charts.WithTitleOpts(opts.Title{
			Title: "CPU - GPU 计算矩阵乘法性能",
		}),
		charts.WithYAxisOpts(opts.YAxis{
			Name: "耗时(ns)",
			SplitLine: &opts.SplitLine{
				Show: false,
			},
		}),
		charts.WithXAxisOpts(opts.XAxis{
			Name: "阶",
		}),
	)

	// Put data into instance
	line.SetXAxis(list_size).
		AddSeries("Category A", getLineItems(cpu_time)).
		AddSeries("Category B", getLineItems(gpu_time)).
		SetSeriesOptions(charts.WithLineChartOpts(opts.LineChart{Smooth: true}))
	line.Render(w)
}
