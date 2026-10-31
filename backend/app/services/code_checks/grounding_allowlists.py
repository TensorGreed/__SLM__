"""Allowlists for the ``context_grounding`` check: names a language provides on its own (keywords,
builtins, standard-library roots, very common methods), so code using them is never asked to find them in
the prompt context. Kept modest on purpose: a missing entry makes the check stricter, so when a common
standard name turns out to be flagged, add it here.

Every set is keyed by language family: ``c_like`` members (c, cpp, csharp, java, javascript /
typescript, go, rust) also get ``COMMON_METHODS``.
"""

from __future__ import annotations

import builtins as _py_builtins
import sys as _sys


def _words(text: str) -> frozenset[str]:
    return frozenset(text.split())


# Methods / attributes that nearly every standard library has on its strings, lists, maps, streams,
# errors and responses. A tail segment in this set is never flagged.
COMMON_METHODS = _words("""
    get set put add append extend push pop shift unshift remove delete insert clear clone copy update
    keys values items entries length size len count index indexOf find findIndex includes contains has
    join split strip lstrip rstrip trim trimStart trimEnd replace replaceAll format lower upper toLowerCase
    toUpperCase startswith endswith startsWith endsWith substring substr slice splice concat reverse sort
    sorted map filter reduce forEach some every flat flatMap first last min max sum
    toString equals hashCode compareTo valueOf parse stringify encode decode
    open close read write readline readlines flush seek send recv connect listen accept bind call apply
    then catch finally resolve reject await wait notify run start stop shutdown cancel done result
    json text data body status code message error errors name id type value key args kwargs
    isEmpty is_empty empty iter next hasNext unwrap expect ok err is_ok is_err as_str as_ref to_string
    to_owned into from new default build builder with_capacity capacity
    Error String Close Read Write Len Name Value Key Err
    lock unlock acquire release submit execute push_back emplace_back pop_back c_str reserve resize
""")

PYTHON_KEYWORDS = _words("""
    False None True and as assert async await break class continue def del elif else except finally for
    from global if import in is lambda nonlocal not or pass raise return try while with yield match case
    self cls super __name__ __file__ __main__ __init__ __dict__ __class__ __doc__ __all__ print
""")

PYTHON_BUILTINS = frozenset(dir(_py_builtins)) | PYTHON_KEYWORDS
PYTHON_STDLIB_ROOTS = frozenset(getattr(_sys, "stdlib_module_names", ())) | _words(
    "os sys re json typing pathlib collections itertools functools logging datetime time math random "
    "subprocess argparse dataclasses abc enum asyncio contextlib io shutil tempfile hashlib hmac base64 uuid "
    "unittest string textwrap struct socket ssl http urllib email csv sqlite3 threading queue signal copy "
    "pickle glob fnmatch platform getpass secrets traceback warnings inspect operator decimal fractions "
    "statistics zipfile tarfile gzip xml html configparser"
)


def _python_methods() -> frozenset[str]:
    names: set[str] = set()
    for t in (str, bytes, list, dict, set, tuple, int, float, object, Exception, type(open), type(iter([]))):
        names.update(n for n in dir(t) if not n.startswith("__"))
    return frozenset(names)


PYTHON_METHODS = _python_methods()

JS_KEYWORDS = _words("""
    break case catch class const continue debugger default delete do else export extends finally for function
    if import in instanceof let new return super switch this throw try typeof var void while with yield async
    await of static get set from as type interface enum implements private protected public readonly
    declare namespace abstract keyof never unknown any boolean number string symbol object undefined null
    true false NaN Infinity arguments require module exports __dirname __filename
""")

JS_GLOBALS = _words("""
    console Math JSON Object Array String Number Boolean Symbol BigInt Date RegExp Error TypeError RangeError
    SyntaxError Promise Map Set WeakMap WeakSet Proxy Reflect Intl ArrayBuffer Uint8Array DataView TextEncoder
    TextDecoder URL URLSearchParams AbortController Buffer process globalThis window document navigator
    fetch setTimeout clearTimeout setInterval clearInterval setImmediate queueMicrotask structuredClone
    parseInt parseFloat isNaN isFinite encodeURIComponent decodeURIComponent encodeURI decodeURI atob btoa
    Record Partial Required Readonly Pick Omit Exclude Extract ReturnType Parameters Awaited NonNullable
    Headers Request Response FormData Blob File Event EventTarget localStorage sessionStorage alert
    describe it test expect beforeEach afterEach beforeAll afterAll jest
""")

NODE_BUILTIN_MODULES = _words("""
    assert buffer child_process cluster crypto dgram dns events fs http http2 https net os path perf_hooks
    process querystring readline stream string_decoder timers tls tty url util v8 vm worker_threads zlib
    async_hooks module inspector diagnostics_channel test
""")

JS_METHODS = _words("""
    log info warn error debug table assign freeze create defineProperty getPrototypeOf isArray from of
    fill at padStart padEnd charAt charCodeAt codePointAt match matchAll search localeCompare normalize repeat
    toFixed toISOString getTime now getFullYear all allSettled race any addEventListener removeEventListener
    querySelector querySelectorAll getElementById createElement appendChild preventDefault stopPropagation
    readFile readFileSync writeFile writeFileSync existsSync mkdirSync promises dirname basename extname
    resolve join env argv exit cwd on once emit pipe end write createServer listen use
""")

JAVA_KEYWORDS = _words("""
    abstract assert boolean break byte case catch char class const continue default do double else enum
    extends final finally float for goto if implements import instanceof int interface long native new
    package private protected public return short static strictfp super switch synchronized this throw
    throws transient try void volatile while var record yield sealed permits true false null
""")

JAVA_TYPES = _words("""
    String Object Integer Long Double Float Boolean Character Byte Short Number Math System Thread Runnable
    Exception RuntimeException IllegalArgumentException IllegalStateException NullPointerException
    IOException UncheckedIOException InterruptedException Throwable Error Override Deprecated
    FunctionalInterface SuppressWarnings StringBuilder List ArrayList LinkedList Map HashMap LinkedHashMap
    TreeMap Set HashSet LinkedHashSet TreeSet Collection Collections Arrays Objects Optional Stream
    Collectors Iterator Iterable Comparable Comparator Function Supplier Consumer Predicate BiFunction
    CompletableFuture Executors ExecutorService TimeUnit Duration Instant LocalDate LocalDateTime UUID
    Path Paths Files File InputStream OutputStream Reader Writer BufferedReader Scanner Random Pattern
    Matcher Charset StandardCharsets Class Void Iterable AutoCloseable Closeable Logger
""")

JAVA_STDLIB_ROOTS = _words("java javax jdk")

JAVA_METHODS = _words("""
    out err println print printf getMessage getClass getName getSimpleName stream collect toList toSet
    toMap groupingBy joining orElse orElseThrow orElseGet isPresent ifPresent ofNullable asList emptyList
    singletonList unmodifiableList requireNonNull currentTimeMillis nanoTime getProperty getenv
    charAt intValue longValue doubleValue booleanValue parseInt parseLong parseDouble trim isBlank
    getKey getValue entrySet keySet putIfAbsent computeIfAbsent getOrDefault containsKey containsValue
    toArray subList addAll removeIf iterator printStackTrace getCause close build append
""")

CSHARP_KEYWORDS = _words("""
    abstract as base bool break byte case catch char checked class const continue decimal default delegate
    do double else enum event explicit extern false finally fixed float for foreach goto if implicit in
    int interface internal is lock long namespace new null object operator out override params private
    protected public readonly ref return sbyte sealed short sizeof stackalloc static string struct switch
    this throw true try typeof uint ulong unchecked unsafe ushort using virtual void volatile while var
    async await dynamic get set value yield record init nameof when where select from orderby group let
""")

CSHARP_TYPES = _words("""
    Console String Int32 Int64 Double Boolean Object Math Convert DateTime TimeSpan Guid Exception
    ArgumentException ArgumentNullException InvalidOperationException NotImplementedException List
    Dictionary HashSet IEnumerable IList IDictionary ICollection Task ValueTask Func Action Enumerable
    StringBuilder Environment File Path Directory Stream StreamReader StreamWriter Encoding Regex Lazy
    Tuple Nullable Array CancellationToken Thread Interlocked IDisposable Attribute Obsolete Serializable
""")

CSHARP_STDLIB_ROOTS = _words("System")

CSHARP_METHODS = _words("""
    WriteLine Write ReadLine ToString Equals GetHashCode GetType Add Remove Contains ContainsKey TryGetValue
    Count Length Select Where First FirstOrDefault Single SingleOrDefault Any All OrderBy OrderByDescending
    GroupBy ToList ToArray ToDictionary Sum Min Max Average Aggregate Skip Take Distinct Concat Join Split
    Trim Replace Substring StartsWith EndsWith IndexOf IsNullOrEmpty IsNullOrWhiteSpace Format Parse
    TryParse Now UtcNow Dispose DisposeAsync ConfigureAwait Result Wait WhenAll FromResult Run Delay
    Message InnerException Value HasValue Keys Values Clear Insert Close Flush
""")

GO_KEYWORDS = _words("""
    break case chan const continue default defer else fallthrough for func go goto if import interface map
    package range return select struct switch type var true false nil iota
    append cap clear close complex copy delete imag len make max min new panic print println real recover
    bool byte complex64 complex128 error float32 float64 int int8 int16 int32 int64 rune string uint uint8
    uint16 uint32 uint64 uintptr any comparable
""")

GO_STDLIB_ROOTS = _words("""
    fmt os io bufio bytes strings strconv errors time context sync atomic net http url json xml csv log slog
    math rand big sort slices maps regexp filepath path reflect testing unicode utf8 hex base64 sha256
    sha1 md5 hmac tls x509 template exec signal ioutil flag heap list ring runtime debug embed fs
""")

RUST_KEYWORDS = _words("""
    as async await break const continue crate dyn else enum extern false fn for if impl in let loop match
    mod move mut pub ref return self Self static struct super trait true type unsafe use where while
    i8 i16 i32 i64 i128 isize u8 u16 u32 u64 u128 usize f32 f64 bool char str
    String Vec Option Some None Result Ok Err Box Rc Arc RefCell Cell Mutex RwLock HashMap HashSet BTreeMap
    BTreeSet VecDeque Cow Default Debug Clone Copy PartialEq Eq Hash PartialOrd Ord Display From Into
    Iterator IntoIterator Send Sync Sized Drop Fn FnMut FnOnce AsRef Deref ToString Error Path PathBuf
    Duration Instant
""")

RUST_STDLIB_ROOTS = _words("std core alloc crate self super Self")

RUST_METHODS = _words("""
    iter iter_mut into_iter collect map filter filter_map and_then or_else unwrap_or unwrap_or_else
    unwrap_or_default ok_or ok_or_else map_err as_bytes as_slice as_mut to_vec chars bytes lines trim
    split_whitespace parse len push push_str extend insert remove get get_mut contains_key entry or_insert
    is_some is_none take replace borrow borrow_mut lock read write clone unwrap expect join spawn
""")

C_KEYWORDS = _words("""
    auto break case char const continue default do double else enum extern float for goto if inline int long
    register restrict return short signed sizeof static struct switch typedef union unsigned void volatile
    while _Bool bool true false NULL nullptr size_t ssize_t ptrdiff_t uint8_t uint16_t uint32_t uint64_t
    int8_t int16_t int32_t int64_t uintptr_t intptr_t FILE EOF errno stdin stdout stderr
    class namespace template typename public private protected virtual override final operator new delete
    this throw try catch using friend explicit mutable constexpr noexcept decltype static_cast
    dynamic_cast reinterpret_cast const_cast std main
""")

C_STDLIB_FUNCTIONS = _words("""
    printf fprintf sprintf snprintf vprintf vfprintf vsnprintf scanf fscanf sscanf puts fputs fgets getchar
    putchar fputc fgetc fopen fclose fread fwrite fflush fseek ftell rewind feof ferror perror remove rename
    tmpfile malloc calloc realloc free exit abort atexit atoi atol atof strtol strtoul strtod getenv system
    qsort bsearch abs labs rand srand memcpy memmove memset memcmp memchr strlen strcpy strncpy strcat
    strncat strcmp strncmp strchr strrchr strstr strtok strdup strndup strerror isalpha isdigit isspace
    isalnum isupper islower toupper tolower time clock difftime mktime localtime gmtime strftime sqrt pow
    sin cos tan floor ceil fabs log exp round assert open close read write lseek unlink sleep usleep fork
    execvp waitpid getpid pipe dup2 socket bind listen accept connect send recv htons htonl ntohs ntohl
    pthread_create pthread_join pthread_mutex_lock pthread_mutex_unlock signal raise setjmp longjmp
    va_start va_end va_arg offsetof
""")

CPP_STD_NAMES = _words("""
    cout cin cerr endl string vector map unordered_map set unordered_set list deque array pair tuple
    make_pair make_tuple make_shared make_unique shared_ptr unique_ptr weak_ptr optional variant function
    thread mutex lock_guard unique_lock move forward swap sort find begin end size to_string stoi stod
    getline runtime_error exception invalid_argument out_of_range ostream istream stringstream ifstream
    ofstream min max accumulate transform
""")

C_STANDARD_HEADERS = _words("""
    assert.h complex.h ctype.h errno.h fenv.h float.h inttypes.h iso646.h limits.h locale.h math.h setjmp.h
    signal.h stdalign.h stdarg.h stdatomic.h stdbool.h stddef.h stdint.h stdio.h stdlib.h stdnoreturn.h
    string.h tgmath.h threads.h time.h uchar.h wchar.h wctype.h unistd.h fcntl.h pthread.h dirent.h
    netdb.h poll.h sched.h semaphore.h strings.h syslog.h termios.h dlfcn.h getopt.h
    algorithm any array atomic bitset cassert cctype cerrno cfloat chrono climits cmath condition_variable
    cstddef cstdint cstdio cstdlib cstring ctime deque exception filesystem fstream functional future
    initializer_list iomanip ios iosfwd iostream istream iterator limits list map memory mutex numeric
    optional ostream queue random ratio regex set shared_mutex sstream stack stdexcept string string_view
    system_error thread tuple type_traits typeinfo unordered_map unordered_set utility variant vector span
""")
C_STANDARD_HEADER_DIRS = ("sys/", "netinet/", "arpa/", "bits/", "linux/")

SHELL_BUILTINS = _words("""
    if then else elif fi for while until do done case esac in function select time return break continue
    local export readonly declare typeset unset shift set source alias unalias eval exec exit trap wait
    read echo printf test true false cd pwd pushd popd dirs let getopts command type hash builtin enable
    help history jobs fg bg kill disown suspend umask ulimit shopt complete mapfile readarray caller logout
    times sudo env nohup xargs nice timeout
""")

SHELL_COMMON_COMMANDS = _words("""
    ls cat cp mv rm mkdir rmdir touch chmod chown chgrp ln find grep egrep fgrep sed awk cut sort uniq head
    tail wc tr tee diff cmp file stat du df mount umount ps top free uptime whoami id groups hostname uname
    date sleep basename dirname realpath readlink mktemp tar gzip gunzip zip unzip xz bzip2 curl wget ssh scp
    rsync git make cmake gcc cc python python3 pip pip3 node npm npx java go cargo rustc docker jq yq
    openssl base64 sha256sum md5sum gpg systemctl service journalctl crontab useradd usermod groupadd
    passwd su apt apt-get yum dnf apk brew which whereis less more nano vi vim tput clear seq yes logger
    nc netstat ss ip ifconfig ping dig nslookup host traceroute lsof kill killall pkill pgrep install
    getent sh bash zsh envsubst column paste join split shuf comm fold expand nl od xxd hexdump strings
    iconv watch flock sync tac rev bc expr
""")

HCL_KEYWORDS = _words("""
    resource data variable output locals module provider terraform required_providers backend count for_each
    depends_on lifecycle provisioner connection dynamic content each var local self path null true false
    for in if moved import check removed source version description type default sensitive validation
    condition error_message nullable value key create_before_destroy prevent_destroy ignore_changes
    replace_triggered_by precondition postcondition iterator labels
    string number bool list map set object tuple any
""")

HCL_FUNCTIONS = _words("""
    abs ceil floor log max min parseint pow signum chomp endswith format formatlist indent join lower
    regex regexall replace split startswith strcontains strrev substr title trim trimprefix trimsuffix
    trimspace upper alltrue anytrue chunklist coalesce coalescelist compact concat contains distinct
    element flatten index keys length lookup matchkeys merge one range reverse setintersection setproduct
    setsubtract setunion slice sort sum transpose values zipmap base64decode base64encode base64gzip
    csvdecode jsondecode jsonencode textdecodebase64 textencodebase64 urlencode yamldecode yamlencode
    abspath dirname pathexpand basename file fileexists fileset filebase64 templatefile formatdate
    timeadd timecmp timestamp plantimestamp base64sha256 base64sha512 bcrypt filebase64sha256 filemd5
    filesha1 filesha256 filesha512 md5 rsadecrypt sha1 sha256 sha512 uuid uuidv5 cidrhost cidrnetmask
    cidrsubnet cidrsubnets can issensitive nonsensitive sensitive tobool tolist tomap tonumber toset
    tostring try type ephemeralasnull
""")

HCL_COMMON_ATTRIBUTES = _words("id name tags count index key value")

# YAML: generic keys so common across tools that no context needs to show them, plus parent keys whose
# children are names the author picks (labels, variables, environment...): those children are never checked.
YAML_COMMON_KEYS = _words("""
    name id type kind version description value values key items item enabled default path paths file files
    mode state src dest url host port ports image command args env environment labels tags metadata spec
    data steps run uses with jobs script stages services volumes when vars variables include includes
    options settings config configuration properties parameters params inputs outputs output input
    title summary content text message list map default_value required depends_on dependencies
""")

YAML_FREE_FORM_PARENTS = _words("""
    labels annotations env environment variables vars data stringData matchLabels selector defaults
    parameters params args properties headers options settings config values extra_vars outputs inputs
    secrets with tags definitions components schemas services volumes networks jobs
""")
