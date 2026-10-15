/*
 * script.js — Car Price Multiple Linear Regression (client-side ML core)
 * ---------------------------------------------------------------------
 * Responsibilities (FEAT-001):
 *   1. Load + parse 01_car_prices.csv at runtime with PapaParse.
 *   2. Auto-identify the target column (name contains 'price') and the
 *      remaining numeric columns as features.
 *   3. Standardize features (z-score) for numerical stability.
 *   4. Fit Multiple Linear Regression via the ridge-regularized normal
 *      equation (X^T X + lambda*I)^-1 X^T y, with the intercept NOT
 *      penalized. Ridge is required because `year` and `age_years` are
 *      near-collinear (age ~= 2026 - year), which makes a naive solve
 *      singular.
 *   5. Compute R^2 on the training data.
 *   6. Expose a documented interface for the UI (FEAT-002) to consume:
 *        window.CarModel = { ready, featureNames, predict, r2 }
 *
 * The DOM wiring lives in the UI feature; this file deliberately keeps
 * presentation concerns out and only publishes a clean model API.
 */
(function () {
  'use strict';

  // ----- Configuration ------------------------------------------------
  var CSV_URL = '01_car_prices.csv';
  var RIDGE_LAMBDA = 1e-2; // small ridge penalty; intercept excluded

  // ----- Linear-algebra helpers (pure JS) -----------------------------

  // Multiply A (m x n) by B (n x p) -> (m x p).
  function matMul(A, B) {
    var m = A.length;
    var n = B.length;
    var p = B[0].length;
    var C = [];
    for (var i = 0; i < m; i++) {
      C[i] = new Array(p).fill(0);
      for (var k = 0; k < n; k++) {
        var a = A[i][k];
        if (a === 0) continue;
        for (var j = 0; j < p; j++) {
          C[i][j] += a * B[k][j];
        }
      }
    }
    return C;
  }

  // Transpose A (m x n) -> (n x m).
  function transpose(A) {
    var m = A.length;
    var n = A[0].length;
    var T = [];
    for (var j = 0; j < n; j++) {
      T[j] = new Array(m);
      for (var i = 0; i < m; i++) {
        T[j][i] = A[i][j];
      }
    }
    return T;
  }

  // Multiply matrix A (m x n) by vector v (length n) -> vector (length m).
  function matVec(A, v) {
    var out = [];
    for (var i = 0; i < A.length; i++) {
      var s = 0;
      var row = A[i];
      for (var j = 0; j < row.length; j++) {
        s += row[j] * v[j];
      }
      out[i] = s;
    }
    return out;
  }

  // Invert a square matrix using Gauss-Jordan elimination with partial
  // pivoting. Returns null if the matrix is singular to working precision.
  function invert(M) {
    var n = M.length;
    // Build augmented [M | I].
    var A = [];
    for (var i = 0; i < n; i++) {
      A[i] = M[i].slice();
      for (var j = 0; j < n; j++) {
        A[i].push(i === j ? 1 : 0);
      }
    }

    for (var col = 0; col < n; col++) {
      // Partial pivot: find row with largest absolute value in this column.
      var pivotRow = col;
      var maxAbs = Math.abs(A[col][col]);
      for (var r = col + 1; r < n; r++) {
        var v = Math.abs(A[r][col]);
        if (v > maxAbs) {
          maxAbs = v;
          pivotRow = r;
        }
      }
      if (maxAbs < 1e-12) {
        return null; // singular
      }
      if (pivotRow !== col) {
        var tmp = A[col];
        A[col] = A[pivotRow];
        A[pivotRow] = tmp;
      }

      // Normalize pivot row.
      var pivot = A[col][col];
      for (var c = 0; c < 2 * n; c++) {
        A[col][c] /= pivot;
      }

      // Eliminate this column from every other row.
      for (var rr = 0; rr < n; rr++) {
        if (rr === col) continue;
        var factor = A[rr][col];
        if (factor === 0) continue;
        for (var cc = 0; cc < 2 * n; cc++) {
          A[rr][cc] -= factor * A[col][cc];
        }
      }
    }

    // Extract the right half (the inverse).
    var inv = [];
    for (var k = 0; k < n; k++) {
      inv[k] = A[k].slice(n, 2 * n);
    }
    return inv;
  }

  // ----- Model state ---------------------------------------------------
  var model = {
    featureNames: [], // ordered feature column names
    targetName: null, // target column name
    means: [], // per-feature mean (for standardization)
    stds: [], // per-feature std (guarded against 0)
    weights: [], // standardized-feature coefficients (length = #features)
    intercept: 0, // bias term
    r2: null
  };

  // ----- Column auto-identification -----------------------------------

  // Decide which columns are numeric features and which is the target.
  function identifyColumns(rows) {
    var headers = Object.keys(rows[0]);

    // Keep only columns that are numeric in every (non-null) row.
    var numericCols = headers.filter(function (h) {
      return rows.every(function (row) {
        var val = row[h];
        return typeof val === 'number' && isFinite(val);
      });
    });

    // Target: first numeric column whose name contains 'price'.
    var target = numericCols.find(function (h) {
      return /price/i.test(h);
    });
    if (!target) {
      // Fallback: use the last numeric column as the target.
      target = numericCols[numericCols.length - 1];
    }

    var features = numericCols.filter(function (h) {
      return h !== target;
    });

    return { features: features, target: target };
  }

  // ----- Training ------------------------------------------------------

  function trainModel(rows) {
    var cols = identifyColumns(rows);
    model.featureNames = cols.features;
    model.targetName = cols.target;

    var n = rows.length;
    var p = cols.features.length;

    // Raw feature matrix + target vector.
    var rawX = [];
    var y = [];
    for (var i = 0; i < n; i++) {
      var featRow = [];
      for (var j = 0; j < p; j++) {
        featRow.push(rows[i][cols.features[j]]);
      }
      rawX.push(featRow);
      y.push(rows[i][cols.target]);
    }

    // Standardization params (z-score), guarding std == 0.
    model.means = new Array(p).fill(0);
    model.stds = new Array(p).fill(1);
    for (var c = 0; c < p; c++) {
      var mean = 0;
      for (var r = 0; r < n; r++) mean += rawX[r][c];
      mean /= n;
      var variance = 0;
      for (var r2i = 0; r2i < n; r2i++) {
        var d = rawX[r2i][c] - mean;
        variance += d * d;
      }
      variance /= n;
      var std = Math.sqrt(variance);
      if (!isFinite(std) || std < 1e-12) std = 1; // guard constant columns
      model.means[c] = mean;
      model.stds[c] = std;
    }

    // Standardized design matrix with a leading intercept column of 1s.
    // Column 0 = intercept; columns 1..p = standardized features.
    var X = [];
    for (var ri = 0; ri < n; ri++) {
      var row = [1];
      for (var cj = 0; cj < p; cj++) {
        row.push((rawX[ri][cj] - model.means[cj]) / model.stds[cj]);
      }
      X.push(row);
    }

    // Ridge normal equation: beta = (X^T X + lambda*I')^-1 X^T y, where
    // I' is the identity but with the intercept entry (0,0) set to 0 so
    // the bias term is NOT penalized.
    var Xt = transpose(X);
    var XtX = matMul(Xt, X); // (p+1) x (p+1)
    for (var k = 1; k < XtX.length; k++) {
      XtX[k][k] += RIDGE_LAMBDA;
    }
    var Xty = matVec(Xt, y); // length p+1

    var inv = invert(XtX);
    if (!inv) {
      throw new Error('Design matrix is singular even after ridge regularization.');
    }
    var beta = matVec(inv, Xty); // length p+1

    model.intercept = beta[0];
    model.weights = beta.slice(1);

    // Guard against any non-finite coefficients.
    var allFinite = isFinite(model.intercept) && model.weights.every(isFinite);
    if (!allFinite) {
      throw new Error('Training produced non-finite coefficients.');
    }

    // R^2 on the training data.
    model.r2 = computeR2(X, y, beta);

    return model;
  }

  // Coefficient of determination for the fitted model.
  function computeR2(X, y, beta) {
    var n = y.length;
    var meanY = 0;
    for (var i = 0; i < n; i++) meanY += y[i];
    meanY /= n;

    var ssRes = 0;
    var ssTot = 0;
    for (var r = 0; r < n; r++) {
      var pred = 0;
      for (var c = 0; c < X[r].length; c++) {
        pred += X[r][c] * beta[c];
      }
      var resid = y[r] - pred;
      ssRes += resid * resid;
      var dev = y[r] - meanY;
      ssTot += dev * dev;
    }
    if (ssTot < 1e-12) return 0;
    return 1 - ssRes / ssTot;
  }

  // ----- Prediction ----------------------------------------------------

  // Predict price for a raw input object keyed by feature column name.
  // Standardizes with the stored means/stds, applies the learned weights,
  // and returns the price in original units (price_mln_uzs).
  function predict(inputObject) {
    if (!model.featureNames.length) {
      throw new Error('Model is not trained yet.');
    }
    var yhat = model.intercept;
    for (var j = 0; j < model.featureNames.length; j++) {
      var name = model.featureNames[j];
      var raw = inputObject[name];
      var num = typeof raw === 'number' ? raw : parseFloat(raw);
      if (!isFinite(num)) {
        throw new Error('Missing or invalid value for feature: ' + name);
      }
      var z = (num - model.means[j]) / model.stds[j];
      yhat += z * model.weights[j];
    }
    return yhat;
  }

  // ----- CSV load + bootstrap -----------------------------------------

  function loadAndTrain() {
    return new Promise(function (resolve, reject) {
      if (typeof Papa === 'undefined') {
        reject(new Error('PapaParse is not loaded. Include it via CDN before script.js.'));
        return;
      }

      fetch(CSV_URL)
        .then(function (resp) {
          if (!resp.ok) {
            throw new Error('HTTP ' + resp.status + ' while fetching ' + CSV_URL);
          }
          return resp.text();
        })
        .then(function (csvText) {
          var parsed = Papa.parse(csvText, {
            header: true,
            dynamicTyping: true,
            skipEmptyLines: true
          });
          if (!parsed.data || !parsed.data.length) {
            throw new Error('CSV parsed to zero rows.');
          }
          trainModel(parsed.data);
          console.log(
            '[CarModel] Model trained. Features: [' +
              model.featureNames.join(', ') +
              '] -> ' +
              model.targetName +
              '. R^2 = ' +
              model.r2.toFixed(4)
          );
          resolve({
            featureNames: model.featureNames.slice(),
            targetName: model.targetName,
            r2: model.r2
          });
        })
        .catch(function (err) {
          var msg =
            'Failed to load/parse ' +
            CSV_URL +
            '. The app must be served over HTTP (e.g. `python3 -m http.server 8000`); ' +
            'opening index.html via file:// will fail due to CORS. Original error: ' +
            err.message;
          console.error('[CarModel] ' + msg);
          reject(new Error(msg));
        });
    });
  }

  // ----- Public interface ---------------------------------------------
  var readyPromise = loadAndTrain();

  window.CarModel = {
    // Resolves once the CSV is loaded and the model is trained; rejects
    // with a clear message (HTTP serving requirement) on failure.
    ready: readyPromise,
    // Ordered feature names, derived from the CSV header. Populated after
    // `ready` resolves; the UI builds its form from this list.
    get featureNames() {
      return model.featureNames.slice();
    },
    // Target column name (price_mln_uzs).
    get targetName() {
      return model.targetName;
    },
    // R^2 on training data (null until trained).
    get r2() {
      return model.r2;
    },
    // Predict price (mln UZS) from a raw feature object.
    predict: predict
  };
})();
