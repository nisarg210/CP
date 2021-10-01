#include <bits/stdc++.h>
#include <cmath>

using namespace std;

#define ull unsigned long long int
#define ll long long int
#define lmin LONG_MIN
#define lmax LONG_MAX
#define imin ll_MIN
#define imax ll_MAX
#define loop(i, a, b) for (int i = a; i < b; ++i)
#define loopr(i, a, b) for (int i = a; i >= b; --i)
#define loops(i, a, b, step) for (int i = a; i < b; i += step)
#define looprs(i, a, b, step) for (int i = a; i >= b; i -= step)
#define mset(f, a) memset(f, a, sizeof(f))
#define P pair
#define PLL pair<long long, long long>
#define PII pair<int, int>
#define PUU pair<unsigned long long int, unsigned long long int>
#define L list
#define V vector
#define D deque
#define ST set
#define MS multiset
#define UM unordered_map
#define mp make_pair
#define pb push_back
#define pf push_front
#define MM multimap
#define F first
#define S second
#define IT iterator
#define RIT reverse_iterator
#define fastio                        \
    ios_base::sync_with_stdio(false); \
    cin.tie(NULL);                    \
    cout.tie(NULL);
#define all(a) a.begin(), a.end()
#define MAXN 1000008
#define level 18

ll M = 1e9 + 7;
ll fact[300000 + 7];
ll sieve[1000000 + 6];

ll power(ll x, ll y)
{
    if (y == 0)
        return 1;

    ll t = power(x, y / 2);

    t = ((t % M) * (t % M)) % M;

    if (y % 2 == 1)
        t = ((t % M) * (x % M)) % M;

    return t;
}

// max of a and b.
ll max(ll a, ll b)
{
    if (a > b)
        return a;
    return b;
}

// min of a and b.
ll min(ll a, ll b)
{
    if (a > b)
        return b;

    return a;
}

bool sortinrev(const pair<ll, ll> &a,
               const pair<ll, ll> &b)
{
    return (a.first > b.first);
}

// primality test.
ll isPrime(ll x)
{
    vector<ll> v;
    ll i;

    for (i = 1; i * i <= x; i++)
    {
        if (x % i == 0)
        {
            if (i * i == x)
                v.push_back(i);
            else
            {
                v.push_back(i);
                v.push_back(x / i);
            }
        }
    }

    return v.size() == 2;
}

ll binarySearch(vector<ll> arr, ll l, ll r, ll x)
{
    if (r >= l)
    {
        ll mid = l + (r - l) / 2;

        // If the element is present at the middle
        // itself
        if (arr[mid] == x)
            return mid;

        // If element is smaller than mid, then
        // it can only be present in left subarray
        if (arr[mid] > x)
            return binarySearch(arr, l, mid - 1, x);

        // Else the element can only be present
        // in right subarray
        return binarySearch(arr, mid + 1, r, x);
    }

    // We reach here when element is not
    // present in array
    return -1;
}

ll gcd(ll a, ll b)
{
    if (b == 0)
        return a;
    return gcd(b, a % b);
}

ll phi(ll n)
{
    ll result = n;

    for (ll p = 2; p * p <= n; ++p)
    {
        if (n % p == 0)
        {
            while (n % p == 0)
                n /= p;
            result -= result / p;
        }
    }
    if (n > 1)
        result -= result / n;
    return result;
}

void dfs(ll source, ll visited[], vector<ll> graph[], ll l)
{
    visited[source] = l;
    ll i;

    for (i = 0; i < graph[source].size(); i++)
    {
        if (visited[graph[source][i]] == -1)
            dfs(graph[source][i], visited, graph, l);
    }
}

ll day(ll d, ll m, ll y)
{
    static ll t[] = {0, 3, 2, 5, 0, 3,
                     5, 1, 4, 6, 2, 4};
    y -= m < 3;
    return (y + y / 4 - y / 100 +
            y / 400 + t[m - 1] + d) %
           7;
}

bool isPerfectSquare(long double x)
{

    long double sr = sqrt(x);

    return ((sr - floor(sr)) == 0);
}

ll getMid(ll s, ll e) { return s + (e - s) / 2; }

/* A recursive function to get the sum of values in the given range  
    of the array. The following are parameters for this function.  
  
    st --> Poller to segment tree  
    si --> Index of current node in the segment tree. Initially  
            0 is passed as root is always at index 0  
    ss & se --> Starting and ending indexes of the segment represented  
                by current node, i.e., st[si]  
    qs & qe --> Starting and ending indexes of query range */
ll getSumUtil(ll *st, ll ss, ll se, ll qs, ll qe, ll si)
{
    // If segment of this node is a part of given range, then return
    // the sum of the segment
    if (qs <= ss && qe >= se)
        return st[si];

    // If segment of this node is outside the given range
    if (se < qs || ss > qe)
        return 0;

    // If a part of this segment overlaps with the given range
    ll mid = getMid(ss, se);
    return getSumUtil(st, ss, mid, qs, qe, 2 * si + 1) +
           getSumUtil(st, mid + 1, se, qs, qe, 2 * si + 2);
}

/* A recursive function to update the nodes which have the given  
index in their range. The following are parameters  
    st, si, ss and se are same as getSumUtil()  
    i --> index of the element to be updated. This index is  
            in the input array.  
diff --> Value to be added to all nodes which have i in range */
void updateValueUtil(ll *st, ll ss, ll se, ll i, ll diff, ll si)
{
    // Base Case: If the input index lies outside the range of
    // this segment
    if (i < ss || i > se)
        return;

    // If the input index is in range of this node, then update
    // the value of the node and its children
    st[si] = st[si] + diff;
    if (se != ss)
    {
        ll mid = getMid(ss, se);
        updateValueUtil(st, ss, mid, i, diff, 2 * si + 1);
        updateValueUtil(st, mid + 1, se, i, diff, 2 * si + 2);
    }
}

// The function to update a value in input array and segment tree.
// It uses updateValueUtil() to update the value in segment tree
void updateValue(ll arr[], ll *st, ll n, ll i, ll new_val)
{
    // Check for erroneous input index
    if (i < 0 || i > n - 1)
    {
        cout << "Invalid Input";
        return;
    }

    // Get the difference between new value and old value
    ll diff = new_val - arr[i];

    // Update the value in array
    arr[i] = new_val;

    // Update the values of nodes in segment tree
    updateValueUtil(st, 0, n - 1, i, diff, 0);
}

// Return sum of elements in range from index qs (quey start)
// to qe (query end). It mainly uses getSumUtil()
ll getSum(ll *st, ll n, ll qs, ll qe)
{
    // Check for erroneous input values
    if (qs < 0 || qe > n - 1 || qs > qe)
    {
        cout << "Invalid Input";
        return -1;
    }

    return getSumUtil(st, 0, n - 1, qs, qe, 0);
}

// A recursive function that constructs Segment Tree for array[ss..se].
// si is index of current node in segment tree st
ll constructSTUtil(ll arr[], ll ss, ll se, ll *st, ll si)
{
    // If there is one element in array, store it in current node of
    // segment tree and return
    if (ss == se)
    {
        st[si] = arr[ss];
        return arr[ss];
    }

    // If there are more than one elements, then recur for left and
    // right subtrees and store the sum of values in this node
    ll mid = getMid(ss, se);
    st[si] = constructSTUtil(arr, ss, mid, st, si * 2 + 1) +
             constructSTUtil(arr, mid + 1, se, st, si * 2 + 2);
    return st[si];
}

/* Function to construct segment tree from given array. This function  
allocates memory for segment tree and calls constructSTUtil() to  
fill the allocated memory */
ll *constructST(ll arr[], ll n)
{
    // Allocate memory for the segment tree

    //Height of segment tree
    ll x = (ll)(ceil(log2(n)));

    //Maximum size of segment tree
    ll max_size = 2 * (ll)pow(2, x) - 1;

    // Allocate memory
    ll *st = new ll[max_size];

    // Fill the allocated memory st
    constructSTUtil(arr, 0, n - 1, st, 0);

    // Return the constructed segment tree
    return st;
}

bool comp(pair<ll, ll> p1, pair<ll, ll> p2)
{
    return p1.second < p2.second;
}

vector<ll> maxSubArraySum(ll a[], ll size)
{
    ll max_so_far = lmin, max_ending_here = 0,
       start = 0, end = 0, s = 0;

    for (ll i = 0; i < size; i++)
    {
        max_ending_here += a[i];

        if (max_so_far < max_ending_here)
        {
            max_so_far = max_ending_here;
            start = s;
            end = i;
        }

        if (max_ending_here < 0)
        {
            max_ending_here = 0;
            s = i + 1;
        }
    }

    vector<ll> v;
    v.pb(max_so_far);
    v.pb(start);
    v.pb(end);

    return v;
}

vector<ll> smallestSumSubarr(ll arr[], ll n)
{
    ll min_ending_here = 0;

    ll min_so_far = lmax, start = 0, end = 0, s = 0;

    for (ll i = 0; i < n; i++)
    {
        min_ending_here += arr[i];

        if (min_so_far > min_ending_here)
        {
            min_so_far = min_ending_here;
            start = s;
            end = i;
        }

        if (min_ending_here > 0)
        {
            min_ending_here = 0;
            s = i + 1;
        }
    }

    vector<ll> v;
    v.pb(min_so_far);
    v.pb(start);
    v.pb(end);

    return v;
}

void bfs(vector<ll> g[], ll dist[], bool v[], ll u)
{
    queue<ll> q;

    q.push(u);
    v[u] = true;
    dist[u] = 0;

    while (!q.empty())
    {
        ll f = q.front();
        q.pop();

        // Enqueue all adjacent of f and mark them visited
        for (auto i = g[f].begin(); i != g[f].end(); i++)
        {
            if (!v[*i])
            {
                q.push(*i);
                v[*i] = true;
                dist[*i] = dist[f] + 1;
            }
        }
    }
}

class DisjSet
{
    ll *rank, *parent, *gc, n;

public:
    // Constructor to create and
    // initialize sets of n items
    DisjSet(ll n)
    {
        rank = new ll[n];
        parent = new ll[n];
        gc = new ll[n];
        this->n = n;
        makeSet();
    }

    // Creates n single item sets
    void makeSet()
    {
        for (ll i = 0; i < n; i++)
        {
            parent[i] = i;
        }

        for (ll i = 0; i < n; i++)
        {
            rank[i] = i;
            gc[i] = i;
        }
    }

    void initi(ll a[])
    {
        for (ll i = 0; i < n; i++)
            gc[i] = a[i] % M;
    }
    // Finds set of given item x
    ll find(ll x)
    {
        // Finds the representative of the set
        // that x is an element of
        if (parent[x] != x)
        {

            // if x is not the parent of itself
            // Then x is not the representative of
            // his set,
            parent[x] = find(parent[x]);

            // so we recursively call Find on its parent
            // and move i's node directly under the
            // representative of this set
        }

        return parent[x];
    }

    // Do union of two sets represented
    // by x and y.
    bool Union(ll x, ll y)
    {
        // Find current sets of x and y
        ll xset = find(x);
        ll yset = find(y);

        // If they are already in same set
        if (xset == yset)
            return false;

        // Put smaller ranked item under
        // bigger ranked item if ranks are
        // different
        if (rank[xset] < rank[yset])
        {
            parent[xset] = yset;
            gc[yset] = gcd(gc[yset], gc[xset]) % M;
        }
        else if (rank[xset] > rank[yset])
        {
            parent[yset] = xset;
            gc[xset] = gcd(gc[yset], gc[xset]) % M;
        }
        // else {
        //     parent[yset] = xset;
        //     gc[xset]=(gcd(gc[yset],gc[xset]))%M;
        //     rank[xset] = rank[xset] + 1;
        // }
        // If ranks are same, then increment
        // rank.
    }

    ll get(ll x)
    {
        return gc[x] % M;
    }
};

bool isPali(string s)
{
    ll st = 0, ed = s.size() - 1;

    while (st <= ed && s[st] == s[ed])
    {
        st++;
        ed--;
    }

    if (st > ed)
        return true;

    return false;
}

ll findSplitPoint(ll arr[], ll n)
{
    // traverse array element and compute sum
    // of whole array
    ll leftSum = 0;
    for (ll i = 0; i < n; i++)
        leftSum += arr[i];

    // again traverse array and compute right sum
    // and also check left_sum equal to right
    // sum or not
    ll rightSum = 0;
    for (ll i = n - 1; i >= 0; i--)
    {
        // add current element to right_sum
        rightSum += arr[i];

        // exclude current element to the left_sum
        leftSum -= arr[i];

        if (rightSum == leftSum)
            return i;
    }

    // if it is not possible to split array
    // into two parts.
    return -1;
}

// Prints two parts after finding split point using
// findSplitPoint()
void printTwoParts(ll arr[], ll n)
{
    ll splitPoint = findSplitPoint(arr, n);

    if (splitPoint == -1 || splitPoint == n)
    {
        cout << "NO\n";
        return;
    }
    cout << "YES\n";
}

bool findPartiion(ll arr[], ll n)
{
    ll sum = 0;
    ll i, j;

    // Calculate sum of all elements
    for (i = 0; i < n; i++)
        sum += arr[i];

    if (sum % 2 != 0)
        return false;

    bool part[sum / 2 + 1][n + 1];

    // initialize top row as true
    for (i = 0; i <= n; i++)
        part[0][i] = true;

    // initialize leftmost column,
    // except part[0][0], as 0
    for (i = 1; i <= sum / 2; i++)
        part[i][0] = false;

    // Fill the partition table in botton up manner
    for (i = 1; i <= sum / 2; i++)
    {
        for (j = 1; j <= n; j++)
        {
            part[i][j] = part[i][j - 1];
            if (i >= arr[j - 1])
                part[i][j] = part[i][j] ||
                             part[i - arr[j - 1]][j - 1];
        }
    }

    /* // uncomment this part to print table  
    for (i = 0; i <= sum/2; i++)  
    {  
    for (j = 0; j <= n; j++)  
        cout<<part[i][j];  
    cout<<endl;  
    } */

    return part[sum / 2][n];
}

ll computeXOR(const ll n)
{
    // Modulus operator are expensive
    // on most of the computers.
    // n & 3 will be equivalent to n % 4
    // n % 4
    switch (n & 3)
    {

    // If n is a multiple of 4
    case 0:
        return n;

    // If n % 4 gives remainder 1
    case 1:
        return 1;

    // If n % 4 gives remainder 2
    case 2:
        return n + 1;

    // If n % 4 gives remainder 3
    case 3:
        return 0;
    }
}

ll sp[1000007];

// Calculating SPF (Smallest Prime Factor) for every
// number till MAXN.
// Time Complexity : O(nloglogn)
void siev()
{
    sp[1] = 1;
    for (ll i = 2; i < MAXN; i++)
        sp[i] = i;

    for (ll i = 4; i < MAXN; i += 2)
        sp[i] = 2;

    for (ll i = 3; i * i < MAXN; i++)
    {
        if (sp[i] == i)
        {
            // marking SPF for all numbers divisible by i
            for (int j = i * i; j < MAXN; j += i)

                if (sp[j] == j)
                    sp[j] = i;
        }
    }
}

// A O(log n) function returning primefactorization
// by dividing by smallest prime factor at every step

ll a[2][100009];

vector<ll> getFactorization(ll x)
{
    vector<ll> ret;
    while (x != 1)
    {
        ret.push_back(sp[x]);
        x = x / sp[x];
    }
    return ret;
}

ll findlcm(ll arr[], ll n)
{
    // Initialize result
    ll ans = arr[0];

    for (ll i = 1; i < n; i++)
        ans = (((arr[i] * ans)) /
               (gcd(arr[i], ans)));

    return ans;
}
void bfs(vector<ll> v[], ll n, ll parent[])
{
    ll i;

    ll flg[100000] = {0};

    queue<ll> q;
    q.push(n);
    parent[n] = -1;
    while (!q.empty())
    {
        ll u = q.front();
        q.pop();
        flg[u] = 1;
        for (i = 0; i < v[u].size(); i++)
        {
            ll ve = v[u][i];

            if (flg[ve] == 0)
            {
                q.push(ve);
                parent[ve] = u;
            }
        }
        // cout<<endl;
    }
}

ll spf[MAXN];
void sieveMethod()
{
    ll i, j;
    spf[1] = 1;
    for (i = 2; i < MAXN; i++)
        spf[i] = i;

    for (i = 4; i < MAXN; i += 2)
        spf[i] = 2;

    for (i = 3; i * i <= MAXN; i++)
        if (spf[i] == i)
            for (j = i * i; j < MAXN; j += i)
                if (spf[j] == j)
                    spf[j] = i;
}
vector<ll> getFactor(ll x)
{
    vector<ll> ret;
    while (x != 1)
    {
        ret.push_back(spf[x]);
        x = x / spf[x];
    }
    return ret;
}

ll s;
ll poww[1000000 + 9];
ll c[1000000 + 9];

// void dfs(vector<ll>v[],ll sou,ll t,ll ary[],ll m[],ll vi[])
// {

//     ll i;
//     vi[sou]=1;
//     vector<ll>vect=getFactor(ary[sou]);

//     for(i=0;i<vect.size();i++)
//     {
//         c[vect[i]]++;
//     }

//     for(i=0;i<vect.size();i++)
//     {
//         ll y=vect[i];

//         while(i<vect.size() && y==vect[i])
//             i++;
//         i--;

//         t=((t%M)*(poww[m[y]+1]%M))%M;

//         t=((t%M)*(m[y]+c[y]+1)%M)%M;

//         m[y]=m[y]+c[y];
//     }

//     a[s][sou]=t;

//     for(i=0;i<vect.size();i++)
//     {
//         c[vect[i]]--;
//     }

//     for(i=0;i<v[sou].size();i++)
//     {
//         if(vi[v[sou][i]]==0)
//         {
//             dfs(v,v[sou][i],t,ary,m,vi);
//             vi[v[sou][i]]=1;
//         }
//     }

//     for(i=0;i<vect.size();i++)
//     {
//         m[vect[i]]--;
//     }

// }

vector<ll> tree[100000 + 9];
ll depth[100000 + 9];
ll parent[100000 + 9][level];

// pre-compute the depth for each node and their
// first parent(2^0th parent)
// time complexity : O(n)
void dfsUtil(vector<ll> v[], ll s, ll vis[], ll dis[])
{
    vis[s] = 1;

    ll i;

    for (i = 0; i < v[s].size(); i++)
    {
        if (vis[v[s][i]] == 0)
        {
            vis[v[s][i]] = 1;
            dis[v[s][i]] = dis[s] + 1;
            dfsUtil(v, v[s][i], vis, dis);
        }
    }
}

// Dynamic Programming Sparse Matrix Approach
// populating 2^i parent for each node
// Time complexity : O(nlogn)
void precomputeSparseMatrix(ll n)
{
    for (ll i = 1; i < level; i++)
    {
        for (ll node = 1; node <= n; node++)
        {
            if (parent[node][i - 1] != -1)
                parent[node][i] =
                    parent[parent[node][i - 1]][i - 1];
        }
    }
}

// Returning the LCA of u and v
// Time complexity : O(log n)
ll lca(ll u, ll v)
{
    if (depth[v] < depth[u])
        swap(u, v);

    ll diff = depth[v] - depth[u];

    // Step 1 of the pseudocode
    for (int i = 0; i < level; i++)
        if ((diff >> i) & 1)
            v = parent[v][i];

    // now depth[u] == depth[v]
    if (u == v)
        return u;

    // Step 2 of the pseudocode
    for (ll i = level - 1; i >= 0; i--)
        if (parent[u][i] != parent[v][i])
        {
            u = parent[u][i];
            v = parent[v][i];
        }

    return parent[u][0];
}

void dfs(vector<ll> v[], ll s, ll dis[], ll vis[], ll a[], ll su[], ll y)
{
    //cout<<s<<endl;

    if (su[s] == -1)
    {
        a[s] = 0;
    }
    else
    {
        a[s] = su[s] - y;
    }

    y += a[s];

    ll i;
    vis[s] = 1;

    for (i = 0; i < v[s].size(); i++)
    {
        if (vis[v[s][i]] == 0)
        {
            dis[v[s][i]] = dis[s] + 1;
            dfs(v, v[s][i], dis, vis, a, su, y);
        }
    }

    y -= a[s];
}

void numberOfNodes(vector<ll> adj[], ll count1[], ll s, ll e)
{
    vector<ll>::iterator u;
    count1[s] = 1;
    for (u = adj[s].begin(); u != adj[s].end(); u++)
    {

        // condition to omit reverse path
        // path from children to parent
        if (*u == e)
            continue;

        // recursive call for DFS
        numberOfNodes(adj, count1, *u, s);

        // update count[] value of parent using
        // its children
        count1[s] += count1[*u];
    }
}

void findClosest(ll A[], ll B[], ll C[], ll p, ll q, ll r)
{

    ll diff = lmax; // Initialize min diff

    // Initialize result
    ll res_i = 0, res_j = 0, res_k = 0;

    // Traverse arrays
    ll i = 0, j = 0, k = 0;
    while (i < p && j < q && k < r)
    {
        // Find minimum and maximum of current three elements
        ll minimum = min(A[i], min(B[j], C[k]));
        ll maximum = max(A[i], max(B[j], C[k]));

        // Update result if current diff is less than the min
        // diff so far
        if (maximum - minimum < diff)
        {
            res_i = i, res_j = j, res_k = k;
            diff = maximum - minimum;
        }

        // We can't get less than 0 as values are absolute
        if (diff == 0)
            break;

        // Increment index of array with smallest value
        if (A[i] == minimum)
            i++;
        else if (B[j] == minimum)
            j++;
        else
            k++;
    }

    // Print result
    ll x, y, z;

    x = A[res_i];
    y = B[res_j];
    z = C[res_k];

    cout << (x - y) * (x - y) + (y - z) * (y - z) + (z - x) * (z - x) << endl;
}

ll getClosest(ll, ll, ll);

// Returns element closest to target in arr[]
ll findClosest(vector<ll> arr, ll n, ll target)
{
    // Corner cases
    //cout<<"11\n";
    if (target <= arr[0])
        return arr[0];
    if (target >= arr[n - 1])
        return arr[n - 1];

    // Doing binary search
    ll i = 0, j = n, mid = 0;
    while (i < j)
    {
        mid = (i + j) / 2;

        if (arr[mid] == target)
            return arr[mid];

        /* If target is less than array element, 
            then search in left */
        if (target < arr[mid])
        {

            // If target is greater than previous
            // to mid, return closest of two
            if (mid > 0 && target > arr[mid - 1])
                return getClosest(arr[mid - 1],
                                  arr[mid], target);

            /* Repeat for left half */
            j = mid;
        }

        // If target is greater than mid
        else
        {
            if (mid < n - 1 && target < arr[mid + 1])
                return getClosest(arr[mid],
                                  arr[mid + 1], target);
            // update i
            i = mid + 1;
        }
    }

    // Only single element left after search
    return arr[mid];
}

ll getClosest(ll val1, ll val2,
              ll target)
{
    if (target - val1 >= val2 - target)
        return val2;
    else
        return val1;
}

bool com(pair<ll, ll> p1, pair<ll, ll> p2)
{
    return p1.second > p2.second;
}

void output()
{
}

int main(int argc, const char *argv[])
{
    // insert code here...

    fastio
        // #ifndef ONLINE_JUDGE
        // freopen("input.txt", "r", stdin);
        // freopen("output.txt", "w", stdout);
        // #endif

        //sieveMethod();

        ll t;
    t = 1;
    cin >> t;

    while (t--)
        output();

    return 0;
}