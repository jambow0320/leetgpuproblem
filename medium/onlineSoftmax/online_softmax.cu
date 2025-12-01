int main()
{
    float *Q, *K, *V;
    float *output;

    int n;
    float pre_li = 0, pre_mi = -__FLT_MAX__;
    float pre_output = 0;
    for(int i = 0; i < n; ++i)
    {
        float s = Q[i] * K[i];
        float now_mi = s; float now_li = expf(s - now_mi);
        float sv = expf(s - now_mi);
        float new_mi = fmaxf(pre_mi, now_mi);

        float alpha = expf(pre_mi - new_mi);
        float beta = expf(now_mi - new_mi);
        float new_li = now_li * beta + pre_li * alpha;

        pre_output = (pre_output * pre_li * alpha + sv * V[i] * beta) / (new_li);

        pre_li = new_li;
        pre_mi = new_mi;
    }

    *output = pre_output;

    return 0;
}