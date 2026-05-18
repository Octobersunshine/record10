package main

import (
	"fmt"
	"math"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

const epsilon = 1e-12

type PCAResult struct {
	Data          *mat.Dense
	VarianceRatio []float64
}

type IPCA struct {
	nComponents int
	nSamples    int
	mean        []float64
	components  *mat.Dense
	explainedVar []float64
}

func PCA(data *mat.Dense, targetDim int) (*PCAResult, error) {
	rows, cols := data.Dims()
	if targetDim <= 0 || targetDim > cols {
		return nil, fmt.Errorf("目标维度%d无效，应在1-%d之间", targetDim, cols)
	}

	means := make([]float64, cols)
	for j := 0; j < cols; j++ {
		sum := 0.0
		for i := 0; i < rows; i++ {
			sum += data.At(i, j)
		}
		means[j] = sum / float64(rows)
	}

	centered := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			centered.Set(i, j, data.At(i, j)-means[j])
		}
	}

	covMat := mat.NewSymDense(cols, nil)
	stat.CovarianceMatrix(covMat, centered, nil)

	forceSymmetric(covMat)

	var svd mat.SVD
	ok := svd.Factorize(centered, mat.SVDThinU|mat.SVDThinV)
	if !ok {
		return nil, fmt.Errorf("SVD分解失败")
	}

	singularValues := svd.Values(nil)
	V := mat.NewDense(cols, cols, nil)
	svd.VTo(V)

	eigenvalues := make([]float64, cols)
	for i := range singularValues {
		if i < cols {
			eigenvalues[i] = (singularValues[i] * singularValues[i]) / float64(rows-1)
		}
	}

	orthonormalizeColumns(V)

	totalVariance := 0.0
	for _, v := range eigenvalues {
		totalVariance += v
	}
	varianceRatio := make([]float64, targetDim)
	for i := 0; i < targetDim; i++ {
		varianceRatio[i] = eigenvalues[i] / totalVariance
	}

	selectedEigenvectors := mat.NewDense(cols, targetDim, nil)
	for j := 0; j < targetDim; j++ {
		for i := 0; i < cols; i++ {
			selectedEigenvectors.Set(i, j, V.At(i, j))
		}
	}

	result := mat.NewDense(rows, targetDim, nil)
	result.Mul(centered, selectedEigenvectors)

	return &PCAResult{
		Data:          result,
		VarianceRatio: varianceRatio,
	}, nil
}

func forceSymmetric(s *mat.SymDense) {
	n := s.SymmetricDim()
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			val := (s.At(i, j) + s.At(j, i)) / 2.0
			s.SetSym(i, j, val)
		}
	}
}

func orthonormalizeColumns(m *mat.Dense) {
	rows, cols := m.Dims()
	for j := 0; j < cols; j++ {
		norm := 0.0
		for i := 0; i < rows; i++ {
			norm += m.At(i, j) * m.At(i, j)
		}
		norm = math.Sqrt(norm)
		if norm > epsilon {
			for i := 0; i < rows; i++ {
				m.Set(i, j, m.At(i, j)/norm)
			}
		}
		for k := 0; k < j; k++ {
			dot := 0.0
			for i := 0; i < rows; i++ {
				dot += m.At(i, j) * m.At(i, k)
			}
			for i := 0; i < rows; i++ {
				m.Set(i, j, m.At(i, j)-dot*m.At(i, k))
			}
		}
		norm = 0.0
		for i := 0; i < rows; i++ {
			norm += m.At(i, j) * m.At(i, j)
		}
		norm = math.Sqrt(norm)
		if norm > epsilon {
			for i := 0; i < rows; i++ {
				m.Set(i, j, m.At(i, j)/norm)
			}
		}
	}
}

func NewIPCA(nComponents int) *IPCA {
	return &IPCA{
		nComponents: nComponents,
		nSamples:    0,
	}
}

func (ipca *IPCA) PartialFit(X *mat.Dense) error {
	rows, cols := X.Dims()
	
	if ipca.nComponents <= 0 || ipca.nComponents > cols {
		return fmt.Errorf("目标维度%d无效，应在1-%d之间", ipca.nComponents, cols)
	}

	if ipca.nSamples == 0 {
		ipca.mean = make([]float64, cols)
		ipca.components = mat.NewDense(cols, ipca.nComponents, nil)
		ipca.explainedVar = make([]float64, ipca.nComponents)
	}

	newNSamples := ipca.nSamples + rows
	oldMean := make([]float64, cols)
	copy(oldMean, ipca.mean)

	for j := 0; j < cols; j++ {
		sum := 0.0
		for i := 0; i < rows; i++ {
			sum += X.At(i, j)
		}
		ipca.mean[j] = (ipca.mean[j]*float64(ipca.nSamples) + sum) / float64(newNSamples)
	}

	centered := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			centered.Set(i, j, X.At(i, j)-oldMean[j])
		}
	}

	if ipca.nSamples == 0 {
		var svd mat.SVD
		ok := svd.Factorize(centered, mat.SVDThinV)
		if !ok {
			return fmt.Errorf("SVD分解失败")
		}
		singularValues := svd.Values(nil)
		V := mat.NewDense(cols, cols, nil)
		svd.VTo(V)

		for j := 0; j < ipca.nComponents; j++ {
			for i := 0; i < cols; i++ {
				ipca.components.Set(i, j, V.At(i, j))
			}
			if j < len(singularValues) {
				ipca.explainedVar[j] = (singularValues[j] * singularValues[j]) / float64(newNSamples-1)
			}
		}
	} else {
		projection := mat.NewDense(rows, ipca.nComponents, nil)
		projection.Mul(centered, ipca.components)

		reconstructed := mat.NewDense(rows, cols, nil)
		reconstructed.Mul(projection, ipca.components.T())

		residual := mat.NewDense(rows, cols, nil)
		for i := 0; i < rows; i++ {
			for j := 0; j < cols; j++ {
				residual.Set(i, j, centered.At(i, j)-reconstructed.At(i, j))
			}
		}

		for j := 0; j < cols; j++ {
			deltaMean := ipca.mean[j] - oldMean[j]
			for i := 0; i < rows; i++ {
				residual.Set(i, j, residual.At(i, j)+deltaMean)
			}
		}

		residualNorm := mat.NewDense(cols, 1, nil)
		for i := 0; i < cols; i++ {
			sum := 0.0
			for j := 0; j < rows; j++ {
				sum += residual.At(j, i) * residual.At(j, i)
			}
			residualNorm.Set(i, 0, math.Sqrt(sum))
		}

		newComponents := mat.NewDense(cols, ipca.nComponents+1, nil)
		for j := 0; j < ipca.nComponents; j++ {
			for i := 0; i < cols; i++ {
				newComponents.Set(i, j, ipca.components.At(i, j))
			}
		}
		for i := 0; i < cols; i++ {
			if residualNorm.At(i, 0) > epsilon {
				newComponents.Set(i, ipca.nComponents, residualNorm.At(i, 0)/float64(rows))
			}
		}

		orthonormalizeColumns(newComponents)
		
		for j := 0; j < ipca.nComponents; j++ {
			for i := 0; i < cols; i++ {
				ipca.components.Set(i, j, newComponents.At(i, j))
			}
		}

		totalVar := 0.0
		for _, v := range ipca.explainedVar {
			totalVar += v
		}
		if totalVar > epsilon {
			scale := float64(ipca.nSamples) / float64(newNSamples)
			for j := range ipca.explainedVar {
				ipca.explainedVar[j] *= scale
			}
		}
	}

	ipca.nSamples = newNSamples
	return nil
}

func (ipca *IPCA) Transform(X *mat.Dense) (*mat.Dense, error) {
	rows, cols := X.Dims()
	if cols != len(ipca.mean) {
		return nil, fmt.Errorf("数据维度不匹配: 期望%d维，得到%d维", len(ipca.mean), cols)
	}

	centered := mat.NewDense(rows, cols, nil)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			centered.Set(i, j, X.At(i, j)-ipca.mean[j])
		}
	}

	result := mat.NewDense(rows, ipca.nComponents, nil)
	result.Mul(centered, ipca.components)
	return result, nil
}

func (ipca *IPCA) FitTransform(X *mat.Dense) (*mat.Dense, error) {
	err := ipca.PartialFit(X)
	if err != nil {
		return nil, err
	}
	return ipca.Transform(X)
}

func (ipca *IPCA) ExplainedVarianceRatio() []float64 {
	totalVar := 0.0
	for _, v := range ipca.explainedVar {
		totalVar += v
	}
	ratios := make([]float64, len(ipca.explainedVar))
	if totalVar > epsilon {
		for i, v := range ipca.explainedVar {
			ratios[i] = v / totalVar
		}
	}
	return ratios
}

func (ipca *IPCA) NSamples() int {
	return ipca.nSamples
}

func (ipca *IPCA) Components() *mat.Dense {
	return ipca.components
}

func (ipca *IPCA) Mean() []float64 {
	return ipca.mean
}
