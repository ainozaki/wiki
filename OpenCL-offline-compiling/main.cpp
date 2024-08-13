// main.cpp
// g++ -std=gnu++11 -O3 main.cpp -lOpenCL -o main
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
 
#ifdef __APPLE__
#  include <OpenCL/opencl.h>
#else
#  include <CL/cl.h>
#endif
 
 
static constexpr cl_uint kNDefaultPlatformEntry = 16;
static constexpr cl_uint kNDefaultDeviceEntry = 16;
 
 
 
 
/*!
 * @brief アラインメントされたメモリを動的確保する関数
 * @param [in] size       確保するメモリサイズ (単位はbyte)
 * @param [in] alignment  アラインメント (2のべき乗を指定すること)
 * @return  アラインメントし，動的確保されたメモリ領域へのポインタ
 */
template<typename T = void*, typename std::enable_if<std::is_pointer<T>::value, std::nullptr_t>::type = nullptr>
static inline T
alignedMalloc(std::size_t size, std::size_t alignment) noexcept
{
#if defined(_MSC_VER) || defined(__MINGW32__)
  return reinterpret_cast<T>(_aligned_malloc(size, alignment));
#else
  void* p;
  return reinterpret_cast<T>(posix_memalign(&p, alignment, size) == 0 ? p : nullptr);
#endif  // defined(_MSC_VER) || defined(__MINGW32__)
}
 
 
/*!
 * @brief アラインメントされたメモリを解放する関数
 * @param [in] ptr  解放対象のメモリの先頭番地を指すポインタ
 */
static inline void
alignedFree(void* ptr) noexcept
{
#if defined(_MSC_VER) || defined(__MINGW32__)
  _aligned_free(ptr);
#else
  std::free(ptr);
#endif  // defined(_MSC_VER) || defined(__MINGW32__)
}
 
 
/*!
 * @brief std::unique_ptr で利用するアラインされたメモリ用のカスタムデリータ
 */
struct AlignedDeleter
{
  /*!
   * @brief デリート処理を行うオペレータ
   * @param [in,out] p  アラインメントされたメモリ領域へのポインタ
   */
  void
  operator()(void* p) const noexcept
  {
    alignedFree(p);
  }
};
 
 
/*!
 * @brief プラットフォームIDを取得
 * @param [in] nPlatformEntry  取得するプラットフォームID数の上限
 * @return  プラットフォームIDを格納した std::vector
 */
static inline std::vector<cl_platform_id>
getPlatformIds(cl_uint nPlatformEntry = kNDefaultPlatformEntry)
{
  std::vector<cl_platform_id> platformIds(nPlatformEntry);
  cl_uint nPlatform;
  if (clGetPlatformIDs(nPlatformEntry, platformIds.data(), &nPlatform) != CL_SUCCESS) {
    std::cerr << "clGetPlatformIDs() failed" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  platformIds.resize(nPlatform);
  return platformIds;
}
 
 
/*!
 * @brief デバイスIDを取得
 * @param [in] platformId    デバイスIDの取得元のプラットフォームのID
 * @param [in] nDeviceEntry  取得するデバイスID数の上限
 * @param [in] deviceType    取得対象とするデバイス
 * @return デバイスIDを格納した std::vector
 */
static inline std::vector<cl_device_id>
getDeviceIds(const cl_platform_id& platformId, cl_uint nDeviceEntry = kNDefaultDeviceEntry, cl_int deviceType = CL_DEVICE_TYPE_DEFAULT)
{
  std::vector<cl_device_id> deviceIds(nDeviceEntry);
  cl_uint nDevice;
  if (clGetDeviceIDs(platformId, deviceType, nDeviceEntry, deviceIds.data(), &nDevice) != CL_SUCCESS) {
    std::cerr << "clGetDeviceIDs() failed" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  deviceIds.resize(nDevice);
  return deviceIds;
}
 
 
 
/*!
 * @brief カーネル関数へ引数をまとめてセットする関数の実態
 * @param [in] kernel  OpenCLカーネルオブジェクト
 * @param [in] idx     セットする引数のインデックス
 * @param [in] first   セットする引数．可変パラメータから1つだけ取り出したもの
 * @param [in] rest    残りの引数
 * @return OpenCLのエラーコード．エラーが出た時点でエラーコードを返却する．
 */
template<typename First, typename... Rest>
static inline cl_uint
setKernelArgsImpl(const cl_kernel& kernel, int idx, const First& first, const Rest&... rest) noexcept
{
  cl_uint errCode = clSetKernelArg(kernel, idx, sizeof(first), &first);
  return errCode == CL_SUCCESS ? setKernelArgsImpl(kernel, idx + 1, rest...) : errCode;
}
 
 
/*!
 * @brief カーネル関数へ最後の引数をセットする
 * @param [in] kernel  OpenCLカーネルオブジェクト
 * @param [in] idx     引数のインデックス
 * @param [in] last    最後の引数
 * @return  OpenCLのエラーコード
 */
template<typename Last>
static inline cl_uint
setKernelArgsImpl(const cl_kernel& kernel, int idx, const Last& last) noexcept
{
  return clSetKernelArg(kernel, idx, sizeof(last), &last);
}
 
 
/*!
 * @brief カーネル関数へ引数をまとめてセットする
 * @param [in] kernel  OpenCLカーネルオブジェクト
 * @param [in] args    セットする引数群
 * @return OpenCLのエラーコード．エラーが出た時点でエラーコードを返却する．
 */
template<typename... Args>
static inline cl_uint
setKernelArgs(const cl_kernel& kernel, const Args&... args) noexcept
{
  return setKernelArgsImpl(kernel, 0, args...);
}
 
 
 
 
/*!
 * @brief このプログラムのエントリポイント
 * @return 終了ステータス
 */
int
main(int argc, char* argv[])
{
  static constexpr int ALIGN = 4096;
  static constexpr std::size_t N = 65536;
 
  if (argc < 2) {
    std::cerr << "Please specify only one or more source file" << std::endl;
    return EXIT_FAILURE;
  }
 
  // ホストのメモリを確保
  std::unique_ptr<int[], AlignedDeleter> hostX(alignedMalloc<int*>(N * sizeof(int), ALIGN));
  std::unique_ptr<int[], AlignedDeleter> hostY(alignedMalloc<int*>(N * sizeof(int), ALIGN));
  std::unique_ptr<int[], AlignedDeleter> hostZ(alignedMalloc<int*>(N * sizeof(int), ALIGN));
 
  // 初期化
  std::mt19937 mt((std::random_device())());
  for (std::size_t i = 0; i < N; i++) {
    //hostX[i] = static_cast<int>(mt());
    //hostY[i] = static_cast<int>(mt());
    hostX[i] = i;
    hostY[i] = i;
  }
  std::fill_n(hostZ.get(), N, 0.0f);
 
  // プラットフォームを取得
  std::vector<cl_platform_id> platformIds = getPlatformIds(1);
 
  // デバイスを取得
  std::vector<cl_device_id> deviceIds = getDeviceIds(platformIds[0], 1, CL_DEVICE_TYPE_DEFAULT);
 
  // コンテキストを生成
  cl_int errCode;
  std::unique_ptr<std::remove_pointer<cl_context>::type, decltype(&clReleaseContext)> context(
      clCreateContext(nullptr, 1, &deviceIds[0], nullptr, nullptr, &errCode), clReleaseContext);
 
  // コマンドキューを生成
  std::unique_ptr<std::remove_pointer<cl_command_queue>::type, decltype(&clReleaseCommandQueue)> cmdQueue(
      clCreateCommandQueue(context.get(), deviceIds[0], 0, &errCode), clReleaseCommandQueue);
 
  // デバイスが用いるメモリオブジェクトの生成
  std::unique_ptr<std::remove_pointer<cl_mem>::type, decltype(&clReleaseMemObject)> deviceX(
      clCreateBuffer(context.get(), CL_MEM_READ_WRITE, N * sizeof(int), nullptr, &errCode), clReleaseMemObject);
  std::unique_ptr<std::remove_pointer<cl_mem>::type, decltype(&clReleaseMemObject)> deviceY(
      clCreateBuffer(context.get(), CL_MEM_READ_WRITE, N * sizeof(int), nullptr, &errCode), clReleaseMemObject);
  std::unique_ptr<std::remove_pointer<cl_mem>::type, decltype(&clReleaseMemObject)> deviceZ(
      clCreateBuffer(context.get(), CL_MEM_READ_WRITE, N * sizeof(int), nullptr, &errCode), clReleaseMemObject);
 
  // ホストのメモリをデバイスのメモリに転送
  errCode = clEnqueueWriteBuffer(cmdQueue.get(), deviceX.get(), CL_TRUE, 0, N * sizeof(int), hostX.get(), 0, nullptr, nullptr);
  errCode = clEnqueueWriteBuffer(cmdQueue.get(), deviceY.get(), CL_TRUE, 0, N * sizeof(int), hostY.get(), 0, nullptr, nullptr);
  errCode = clEnqueueWriteBuffer(cmdQueue.get(), deviceZ.get(), CL_TRUE, 0, N * sizeof(int), hostZ.get(), 0, nullptr, nullptr);
 
  // コンパイル後のカーネルのバイナリを読み込み
  std::ifstream ifs(argv[1], std::ios::binary);
  if (!ifs.is_open()) {
    std::cerr << "Failed to kernel binary: " << argv[1] << std::endl;
    std::exit(EXIT_FAILURE);
  }
  std::string kernelBin((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
 
  // プログラムオブジェクトの生成
  const unsigned char* kbin = reinterpret_cast<const unsigned char*>(kernelBin.c_str());
  std::size_t kbinSize = kernelBin.size();
  cl_int binStatus;
  std::unique_ptr<std::remove_pointer<cl_program>::type, decltype(&clReleaseProgram)> program(
      clCreateProgramWithBinary(context.get(), 1, &deviceIds[0], &kbinSize, &kbin, &binStatus, &errCode), clReleaseProgram);
 
  // カーネルソースコードのコンパイル (必要な環境もあるらしい?)
  // errCode = clBuildProgram(program.get(), 1, &deviceIds[0], nullptr, nullptr, nullptr);
 
  // カーネルオブジェクトの生成
  std::unique_ptr<std::remove_pointer<cl_kernel>::type, decltype(&clReleaseKernel)> kernel(
      clCreateKernel(program.get(), "vecAdd", &errCode), clReleaseKernel);
 
  // カーネル関数に引数を渡す
  errCode = setKernelArgs(kernel.get(), deviceZ.get(), deviceX.get(), deviceY.get(), static_cast<int>(N));
 
  // カーネルプログラムの実行
  errCode = clEnqueueTask(cmdQueue.get(), kernel.get(), 0, nullptr, nullptr);
 
  // 終了待機等
  errCode = clFlush(cmdQueue.get());
  errCode = clFinish(cmdQueue.get());
 
  // 実行結果をデバイスからホストへコピー
  errCode = clEnqueueReadBuffer(cmdQueue.get(), deviceZ.get(), CL_TRUE, 0, N * sizeof(int), hostZ.get(), 0, nullptr, nullptr);
 
  // 計算結果の確認
  int count = N;
  for (std::size_t i = 0; i < N; i++) {
    if (std::abs(hostX[i] + hostY[i] - hostZ[i]) > 1.0e-5) {
        count -= 1;
        //printf("Failed: host[%ld] %d + %d != %d\n", i, hostX[i], hostY[i], hostZ[i]);
      //return EXIT_FAILURE;
    }
  }
  std::cout << "Test PASSED: " << count << "/" << N << std::endl;
 
  return EXIT_SUCCESS;
}