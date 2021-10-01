#include <bits/stdc++.h>

using namespace std;

#define ll long long int
#define FASTIO                        \
	ios_base::sync_with_stdio(false); \
	cin.tie(NULL);                    \
	cout.tie(NULL) // Never use this statement in interactive questions.
#define FOR1(i, m, n) for (int i = m; i < n; i++)
#define FOR2(i, n, m) for (int i = n - 1; i >= m; i--)
#define v(v) \
	ll v;    \
	cin >> v
#define mod 1000000007

// fastscan.
void fs(ll &number)
{
	bool negative = false;
	register int c;
	number = 0;
	c = getchar();
	if (c == '-')
	{
		negative = true;
		c = getchar();
	}
	for (; (c > 47 && c < 58); c = getchar())
		number = number * 10 + c - 48;
	if (negative)
		number *= -1;
}

//used to get modulus of a large number with no. of digits greater than 20(limit of ll)
ll modOfLargeNo(string s, ll n)
{
	ll res = 0;
	for (ll i = 0; i < s.length(); i++)
	{
		res = (res * 10 + (ll)s[i] - '0') % n;
	}

	return res;
}

//Use emplace_back() instead of push_back for faster executing.

//Number of digits in a number
ll numOfDigits(ll n)
{
	return floor(log10(n)) + 1;
}

//best parity checker algorithm available with time complexity of O(logn)
//return 1 if n has odd parity and return 0 if n has even parity
bool getParity(unsigned int n)
{
	bool parity = 0;
	while (n)
	{
		parity = !parity;
		n = n & (n - 1);
	}

	return parity;
}

//find n when sigma(n) is given
ll findS(ll s)
{
	return (sqrt(8 * s) + 1) / 2;
}

//primality_test
bool isprime(ll n)
{
	if (n <= 1)
		return false;
	if (n <= 3)
		return true;
	if (n % 2 == 0 || n % 3 == 0)
		return false;
	for (ll i = 5; i * i <= n; i = i + 6)
	{
		if (n % i == 0 || n % (i + 2) == 0)
		{
			return false;
		}
	}
	return true;
}

//binary search iterative algorithm with a time complexity of O(logn) and O(1) space taken
int binSearch(ll arr[], ll l, ll r, ll x)
{

	while (l <= r)
	{
		ll m = l + (r - l) / 2;

		if (arr[m] == x)
		{
			return m;
		}

		if (arr[m] < x)
		{
			l = m + 1;
		}
		else
		{
			r = m - 1;
		}
	}
	//the element is not present if we reach here
	return -1;
}

ll gcd(ll a, ll b)
{

	if (b == 0)
		return a;
	return gcd(b, a % b);
}

ll lcm(ll a, ll b)
{
	return (a * b) / gcd(a, b);
}

ll countDivisors(ll n)
{
	ll cnt = 0;
	for (int i = 1; i <= sqrt(n); i++)
	{
		if (n % i == 0)
		{
			if (n / i == i)
				cnt++;
			else
				cnt = cnt + 2;
		}
	}
	return cnt;
}

void dfs(vector<ll> v[], bool visited[], ll current_node)
{
	visited[current_node] = true;
	for (ll nxt_node : v[current_node])
	{
		if (visited[nxt_node] == false)
		{
			dfs(v, visited, nxt_node);
		}
	}
	return;
}

ll nCr(ll n, ll r)
{
	if (r == 0)
		return 1;
	return (n * nCr(n - 1, r - 1)) / r;
}

bool checkisTree(ll degree[], ll n)
{
	ll deg_sum = 0;
	for (ll i = 0; i < n; i++)
		deg_sum += degree[i];
	return (2 * (n - 1) == deg_sum);
}

bool ckmin(int &a, int b) { return b < a ? a = b, true : false; }

bool ckmax(int &a, int b) { return b > a ? a = b, true : false; }

// reading int and long long int fast.
ll read_int()
{
	char r;
	bool start = false, neg = false;
	ll ret = 0;
	while (true)
	{
		r = getchar();
		if ((r - '0' < 0 || r - '0' > 9) && r != '-' && !start)
		{
			continue;
		}
		if ((r - '0' < 0 || r - '0' > 9) && r != '-' && start)
		{
			break;
		}
		if (start)
			ret *= 10;
		start = true;
		if (r == '-')
			neg = true;
		else
			ret += r - '0';
	}
	if (!neg)
		return ret;
	else
		return -ret;
}

int main(int argc, char const *argv[])
{
	FASTIO;
	int t;
	cin >> t;
	while (t--)
	{
		int n;
		cin >> n;
	}
	return 0;
}