package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"strconv"
	"strings"

	"../../kernel"
	"golang.org/x/image/bmp"
)

var edge15x15 = [][]float64{
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},

	{-1, -1, -1, -1, -1, 8, 8, 8, 8, 8, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, 8, 8, 8, 8, 8, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, 8, 8, 8, 8, 8, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, 8, 8, 8, 8, 8, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, 8, 8, 8, 8, 8, -1, -1, -1, -1, -1},

	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
}
var edge12x12 = [][]float64{
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, 8, 8, 8, 8, -1, -1, -1, -1},
	{-1, -1, -1, -1, 8, 8, 8, 8, -1, -1, -1, -1},
	{-1, -1, -1, -1, 8, 8, 8, 8, -1, -1, -1, -1},
	{-1, -1, -1, -1, 8, 8, 8, 8, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
}
var edge99 = [][]float64{
	{-1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, 8, 8, 8, -1, -1, -1},
	{-1, -1, -1, 8, 8, 8, -1, -1, -1},
	{-1, -1, -1, 8, 8, 8, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1},
	{-1, -1, -1, -1, -1, -1, -1, -1, -1},
}
var edge33 = [][]float64{
	{-1, -1, -1},
	{-1, 8, -1},
	{-1, -1, -1},
}
var lxe = [][]float64{
	{1, 0, -1},
	{2, 0, -2},
	{1, 0, -1},
}
var lye = [][]float64{
	{1, 2, 1},
	{0, 0, 0},
	{-1, -2, -1},
}

func main() {
	//convertjpgtobmp("playground.jpg")
	convvscroswithreverse("playground.bmp", lxe, false, true, 1)
}

func convvscroswithreverse(imagename string, karray [][]float64, reverse bool, zeronegs bool, threads int) {

	convolution(imagename, "cvexample", karray, []int{1, 1}, []int{1, 1}, []int{1, 1}, false, zeronegs, true, threads)
	convolution(imagename, "ccexample", karray, []int{1, 1}, []int{1, 1}, []int{1, 1}, false, zeronegs, false, threads)
	convolution(imagename, "cvexamplerev", karray, []int{1, 1}, []int{1, 1}, []int{1, 1}, true, zeronegs, true, threads)
	convolution(imagename, "ccexamplerev", karray, []int{1, 1}, []int{1, 1}, []int{1, 1}, true, zeronegs, false, threads)
}

func getconvpics(imagename string) {
	stride := []int{1, 1}
	//	dilation := []int{1, 1}
	reverse := false
	zeronegs := false
	conv := false
	threads := 16
	/*
		x := 7
		y := 7
		x1 := 3
		y1 := 3
		//	dx, dy := (x-1)/(x1-1), (y-1)/(y1-1)
		//	fmt.Println(bigedge)
	*/
	convolution(imagename, "stride2x2edge33", edge33, []int{2, 2}, []int{1, 1}, []int{1, 1}, reverse, zeronegs, conv, threads)
	convolution(imagename, "stride3x3edge33", edge33, []int{3, 3}, []int{1, 1}, []int{1, 1}, reverse, zeronegs, conv, threads)
	//	convolution(imagename, "edge99", edge99, stride, []int{1, 1}, []int{4, 4}, reverse, zeronegs, threads)
	//convolution(imagename, "edge12x12", edge12x12, stride, []int{1, 1}, []int{5, 5}, reverse, zeronegs, threads)
	convolution(imagename, "edge15x15", edge15x15, stride, []int{1, 1}, []int{7, 7}, reverse, zeronegs, conv, threads)
	convolution(imagename, "dilation11like33", edge33, stride, []int{1, 1}, []int{1, 1}, reverse, zeronegs, conv, 1)
	//convolution(imagename, "dilation44,like99,", edge33, stride, []int{4, 4}, []int{4, 4}, reverse, zeronegs, 1)
	//	convolution(imagename, "dilation55,like11x11,", edge33, stride, []int{5, 5}, []int{5, 5}, reverse, zeronegs, 1)
	convolution(imagename, "dilation77like15x15,", edge33, stride, []int{7, 7}, []int{7, 7}, reverse, zeronegs, conv, 1)
}

//	convertjpgtobmp("spideronclover256.jpg")
//convolution("spideredges22d11p11.bmp", "reversespiders22d11p11.bmp", firstkernel, stride, dilation, padding, true)

func createname(originalname string, w [][]float64, s, d, p []int, reverse, zeroneg bool) string {
	var newname string
	var suffix string
	if strings.Contains(originalname, "bmp") {
		newname = strings.TrimSuffix(originalname, ".bmp")
		suffix = ".bmp"

	} else if strings.Contains(originalname, "jpg") {
		newname = strings.TrimSuffix(originalname, ".jpg")
		suffix = ".jpg"
	} else if strings.Contains(originalname, "png") {
		newname = strings.TrimSuffix(originalname, ".png")
		suffix = ".png"
	}
	var rv = "reverse"
	var zn = "zeroneg"
	if reverse {
		rv = rv + "1"
	} else {
		rv = rv + "0"
	}
	if zeroneg {
		zn = zn + "1"
	} else {
		zn = zn + "0"
	}
	return newname + wtostring(w) + ptos(s, "s") + ptos(d, "d") + ptos(p, "p") + rv + zn + suffix
}
func ptos(x []int, label string) string {
	for i := range x {
		label = label + strconv.Itoa(x[i])
	}
	return label
}
func wtostring(x [][]float64) string {
	return "w" + strconv.Itoa(len(x)) + strconv.Itoa(len(x[0]))
}
func makeedge(y, x int) [][]float64 {
	positive := float64(x*y) - 1.0
	edge := make([][]float64, y)
	for i := range edge {
		edge[i] = make([]float64, x)
		for j := range edge[i] {
			edge[i][j] = -1
		}
	}

	mx := (x) / 2
	my := (y) / 2
	edge[my][mx] = positive
	return edge
}

func convolution(imagelocation string, newname string, k [][]float64, s, d, p []int, reverse, zeronegatives bool, conv bool, threads int) {

	newname = newname + createname(imagelocation, k, s, d, p, reverse, zeronegatives)

	file, err := os.Open(imagelocation)
	defer file.Close()
	if err != nil {
		fmt.Println("ON file")
		panic(err)
	}
	var img image.Image
	if strings.Contains(imagelocation, "bmp") {
		img, err = bmp.Decode(file)
		if err != nil {
			fmt.Println("on bmp encode")
			panic(err)
		}
	} else if strings.Contains(imagelocation, "jpg") {
		img, err = jpeg.Decode(file)
		if err != nil {
			fmt.Println("on jpeg encode")
			panic(err)
		}
	} else if strings.Contains(imagelocation, "png") {
		img, err = png.Decode(file)
		if err != nil {
			fmt.Println("on pgn encode")
			panic(err)
		}
	} else {
		panic("not supported file extention for decode. Must be bmp,jpg,or png")
	}
	var convimg image.Image
	if reverse {
		if conv {
			convimg = kernel.InverseConvolution(img, k, s, d, p, zeronegatives)
		} else {
			convimg = kernel.InverseCCorelation(img, k, s, d, p, zeronegatives)
		}

	} else {
		if conv {
			convimg = kernel.Convolution(img, k, s, d, p, zeronegatives, threads)
		} else {
			convimg = kernel.CrossCorelation(img, k, s, d, p, zeronegatives, threads)
		}

	}

	newfile, err := os.Create(newname)
	defer newfile.Close()
	if err != nil {
		fmt.Println("ON newfile")
		panic(err)
	}
	if strings.Contains(newname, "bmp") {
		err = bmp.Encode(newfile, convimg)
		if err != nil {
			fmt.Println("on bmp.Encode")
			panic(err)
		}
	} else if strings.Contains(newname, "jpg") {
		err = jpeg.Encode(newfile, convimg, nil)
		if err != nil {
			fmt.Println("on jpeg.Encode")
			panic(err)
		}
	} else if strings.Contains(newname, "png") {
		err = png.Encode(newfile, convimg)
		if err != nil {
			fmt.Println("on png.Encode")
			panic(err)
		}
	} else {
		panic("not supported file extention for encode. Must be bmp,jpg,or png")
	}

}
func convertjpgtobmp(imagelocation string) {
	file, err := os.Open(imagelocation)
	defer file.Close()
	if err != nil {
		fmt.Println("ON file")
		panic(err)
	}
	img, err := jpeg.Decode(file)
	if err != nil {
		fmt.Println("on jpeg decode")
		panic(err)
	}
	newname := strings.TrimSuffix(imagelocation, "jpg") + "bmp"
	newfile, err := os.Create(newname)
	defer newfile.Close()
	if err != nil {
		fmt.Println("ON newfile")
		panic(err)
	}
	err = bmp.Encode(newfile, img)
	if err != nil {
		fmt.Println("on bmp.Encode")
		panic(err)
	}
}
