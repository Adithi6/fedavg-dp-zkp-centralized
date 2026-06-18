pragma circom 2.0.0;

template Num2Bits(n) {
    signal input in;
    signal output out[n];
    var lc1 = 0;

    var e2 = 1;
    for (var i = 0; i < n; i++) {
        out[i] <-- (in >> i) & 1;
        out[i] * (out[i] - 1) === 0;
        lc1 += out[i] * e2;
        e2 = e2 + e2;
    }
    lc1 === in;
}

template LessThan(n) {
    assert(n <= 252);
    signal input in[2];
    signal output out;

    component n2b = Num2Bits(n + 1);

    n2b.in <== in[0] + (1 << n) - in[1];

    out <== 1 - n2b.out[n];
}

template UpdateNorm(n) {
    signal input values[n];
    signal input threshold;

    signal squares[n];
    signal partial[n + 1];

    partial[0] <== 0;

    for (var i = 0; i < n; i++) {
        squares[i] <== values[i] * values[i];
        partial[i + 1] <== partial[i] + squares[i];
    }

    signal output sum;
    sum <== partial[n];

    component lessThan = LessThan(64); 
    lessThan.in[0] <== sum;
    lessThan.in[1] <== threshold;
    
    lessThan.out === 1;
}

component main = UpdateNorm(10);