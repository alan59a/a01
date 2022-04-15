// a Simple MNIST loader written in Go
package a01

import (
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

type img struct {
	data  []uint8
	label uint8
	row   int
	col   int
}

// Reads MNIST image data train and test files
// make sure that the names are the original ones ... meaning these:
// train-images.idx3-ubyte
// t10k-images.idx3-ubyte
func ReadImage(path string) ([][]uint8, [][]uint8) {
	// Loading ...

	file1, err := os.Open(path + "/train-images.idx3-ubyte")
	check(err)
	defer file1.Close()

	file2, err := os.Open(path + "/t10k-images.idx3-ubyte")
	check(err)
	defer file2.Close()

	fmt.Println("Image files found.")

	// Checking the magic number ... it's a 4 byte (32 bit) unsigned integer
	// You can find all of these informations in MNIST web page ...
	magic := make([]byte, 4)

	file1.Read(magic)

	fmt.Println(magic)

	if binary.BigEndian.Uint32(magic) != 2051 {
		log.Fatalln("Incorrect train image file. File location: " + file1.Name())
	}

	file2.Read(magic)

	if binary.BigEndian.Uint32(magic) != 2051 {
		log.Fatalln("Incorrect test image file. File location: " + file2.Name())
	}

	// Checking the number of items
	// Big fan of minimalism and also love my ram space so ... i'm going to reuse the magic ...
	// but remember it's not the magic number anymore
	file1.Read(magic)
	len1 := int(binary.BigEndian.Uint32(magic))
	data1 := make([][]uint8, len1)

	// Checking the image size ...
	file1.Read(magic)
	rows := int(binary.BigEndian.Uint32(magic))

	file1.Read(magic)
	cols := int(binary.BigEndian.Uint32(magic))

	for a := range data1 {
		data1[a] = make([]uint8, rows*cols)
	}

	// Now reading the data bytes for file1 one by one ... now each byte is a pixel value actually
	d := make([]byte, 1)

	// Progress bar ... it's nice to have a bar
	bar1 := newBar(len1, "Loading train image files ...")

	for a := 0; a < len(data1); a++ {

		for b := 0; b < len(data1[a]); b++ {
			_, err = file1.Read(d)

			if err != nil {

				if err == io.EOF {
					break
				} else {
					log.Fatalln(err)
				}
			}
			data1[a][b] = d[0]
		}
		bar1.add(1)
	}

	// And again for file2 ...
	file2.Read(magic)
	len2 := int(binary.BigEndian.Uint32(magic))
	data2 := make([][]uint8, len2)

	// Checking the image size ...
	file2.Read(magic)
	rows = len2

	file2.Read(magic)
	cols = len2

	for a := range data2 {
		data2[a] = make([]uint8, rows*cols)
	}

	// Progress bar
	bar2 := newBar(len2, "Loading test image files ...")

	// Now reading the data ...
	for a := 0; a < len(data2); a++ {

		for b := 0; b < len(data2[a]); b++ {
			_, err = file2.Read(d)

			if err != nil {

				if err == io.EOF {
					break
				} else {
					log.Fatalln(err)
				}
			}
			data2[a][b] = d[0]
		}
		bar2.add(1)
	}

	return data1, data2
}

// Reads MNIST label train and test files
// again ... make sure that the names are the original ones ... meaning these:
// train-labels.idx1-ubyte
// t10k-labels.idx1-ubyte
func ReadLabel(path string) ([]uint8, []uint8) {
	file1, err := os.Open(path + "/train-labels.idx1-ubyte")
	check(err)
	defer file1.Close()

	file2, err := os.Open(path + "/t10k-labels.idx1-ubyte")
	check(err)
	defer file2.Close()
	fmt.Println("Label files found.")

	// Checking the magic number ... it's a 4 byte (32 bit) unsigned integer
	magic := make([]byte, 4)

	file1.Read(magic)

	if binary.BigEndian.Uint32(magic) != 2049 {
		log.Fatalln("Incorrect train label file. File location: " + file1.Name())
	}

	file2.Read(magic)

	if binary.BigEndian.Uint32(magic) != 2049 {
		log.Fatalln("Incorrect test label file. File location: " + file2.Name())
	}

	// Checking the number of items
	// Big fan of minimalism ... i'm going to reuse the magic
	file1.Read(magic)
	len1 := int(binary.BigEndian.Uint32(magic))
	data1 := make([]uint8, len1)

	// Now reading the data bytes for file1 one by one ... according to the MNIST docs
	d := make([]byte, 1)

	// It's nice to have a progress indicator ... isn't it?
	bar1 := newBar(len1, "Loading train label files ... ")

	for a := 0; a < len1; a++ {
		_, err = file1.Read(d)
		if err != nil {
			if err == io.EOF {
				break
			} else {
				log.Fatalln(err)
			}
		}
		data1[a] = d[0]
		bar1.add(1)
	}

	// And again for file2 ...
	file2.Read(magic)

	len2 := int(binary.BigEndian.Uint32(magic))
	data2 := make([]uint8, len2)

	// Progress bar
	bar2 := newBar(len2, "Loading test label files ... ")

	for a := 0; a < len2; a++ {

		_, err = file2.Read(d)

		if err != nil {

			if err == io.EOF {
				break
			} else {
				log.Fatalln(err)
			}
		}
		data2[a] = d[0]
		bar2.add(1)
	}

	return data1, data2
}

// Outputs the image data normalized [0, 1] as *mat.Dense for those trying ML with Gonum.
func ConvertImage(data [][]uint8, row, col int) []*mat.Dense {
	mats := make([]*mat.Dense, len(data))
	d := make([]float64, row*col)

	// Always nice to have a progress bar
	bar := newBar(len(data), "Converting to mat.Dense ... ")

	for a := range data {

		for b := range data[a] {
			d[b] = float64(data[a][b]) / 255
		}
		mats[a] = mat.NewDense(row, col, d)
		bar.add(1)
	}

	return mats
}

// Outputs the label data normalized [0 or 1] as *mat.Dense for those trying ML with Gonum.
func ConvertLabel(data []uint8) []*mat.Dense {
	mats := make([]*mat.Dense, len(data))

	// Always nice to have a progress bar
	bar := newBar(len(data), "Converting to mat.Dense ... ")

	for a, b := range data {
		d := make([]float64, 10)
		d[b] = 1
		mats[a] = mat.NewDense(10, 1, d)
		bar.add(1)
	}

	return mats
}

// Since we're doing a LOT of error checking ...
func check(err error) {
	if err != nil {
		log.Fatalln(err)
	}
}

func ToImg(data []uint8, label uint8, row, col int) *img {
	return &img{
		data:  data,
		label: label,
		row:   row,
		col:   col,
	}
}

// Printing the img file for visual effects ... Use "pre" for anything needed before the name
// TO DO ... i  got bored ... still works somehow ...
func (i *img) ShowImage(path, pre string) {
	img := image.NewRGBA(image.Rectangle{
		Min: image.Point{
			X: 0,
			Y: 0,
		},
		Max: image.Point{
			X: i.row,
			Y: i.col,
		},
	})

	for a := 0; a < i.col; a++ {
		for b := 0; b < i.row; b++ {
			img.Set(b, a, color.Gray{
				Y: 255 - i.data[a*i.col+b],
			})
		}
	}

	file, err := os.Create(path + "/" + pre + " - " + strconv.Itoa(int(i.label)) + ".png")
	check(err)
	defer file.Close()

	if err = png.Encode(file, img); err != nil {
		check(err)
	}
}
