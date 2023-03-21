#include <stdexec/execution.hpp>

namespace Kokkos::StdExec {

class inline_scheduler {
  template <class R>
  struct _op {
    [[no_unique_address]] R rec_;
    friend void tag_invoke(stdexec::start_t, _op& op) noexcept try {
      stdexec::set_value((R &&) op.rec_);
    } catch (...) {
      stdexec::set_error((R &&) op.rec_, std::current_exception());
    }
  };

  struct _attrs {
    template <class Tag>
    friend inline_scheduler tag_invoke(
        stdexec::get_completion_scheduler_t<Tag>, _attrs) noexcept {
      return {};
    }
  };

  struct _sender {
    using completion_signatures =
        stdexec::completion_signatures<stdexec::set_value_t(),
                                              stdexec::set_error_t(
                                                  std::exception_ptr)>;

    template <class R>
    friend auto tag_invoke(stdexec::connect_t, _sender, R&& rec) noexcept(
        std::is_nothrow_constructible_v<std::remove_cvref_t<R>, R>)
        -> _op<std::remove_cvref_t<R>> {
      return {(R &&) rec};
    }

    friend _attrs tag_invoke(stdexec::get_env_t, _sender) noexcept {
      return {};
    }
  };

  friend _sender tag_invoke(stdexec::schedule_t,
                            const inline_scheduler&) noexcept {
    return {};
  }

 public:
  inline_scheduler()                                      = default;
  bool operator==(const inline_scheduler&) const noexcept = default;
};

template <class _ReceiverId, std::integral _Shape, class _Fun>
struct bulk_receiver {
  [[no_unique_address]] _Shape __shape_;
  [[no_unique_address]] _Fun __f_;
  [[no_unique_address]] _ReceiverId successor;

  friend void tag_invoke(stdexec::set_value_t, bulk_receiver&& self) noexcept {
    for (_Shape __i{}; __i != self.__shape_; ++__i) {
      self.__f_(__i);
    }
    stdexec::set_value(std::move(self.successor));
  }

  explicit bulk_receiver(_ReceiverId __rcvr, _Shape __shape, _Fun __fun):
      __shape_(__shape), __f_((_Fun &&) __fun), successor(__rcvr) {}
};

template <class _Sender, std::integral _Shape, class _Fun>
struct bulk_sender {
  template <stdexec::receiver _Receiver>
  using __receiver = bulk_receiver<_Receiver, _Shape, _Fun>;

  using is_sender = void;

  [[no_unique_address]] _Sender __sndr_;
  [[no_unique_address]] _Shape __shape_;
  [[no_unique_address]] _Fun __fun_;

  using completion_signatures =
      stdexec::completion_signatures<stdexec::set_value_t(),
                                            stdexec::set_error_t(
                                                std::exception_ptr)>;

  template <stdexec::receiver _Receiver>
  friend auto tag_invoke(stdexec::connect_t, bulk_sender&& __self,
                         _Receiver __rcvr) noexcept {
    return stdexec::connect(
        std::move(__self.__sndr_),
        __receiver<_Receiver>{std::move(__rcvr), __self.__shape_,
                              std::move(__self.__fun_)});
  }

  friend auto tag_invoke(stdexec::get_env_t, const bulk_sender& __self) noexcept {
    return stdexec::get_env(__self.__sndr_);
  }
};

template<class Sender, std::integral Shape, class Fun>
auto tag_invoke(stdexec::bulk_t, inline_scheduler, Sender&& snd, Shape shape, Fun fun) {
  printf("Yay\n");
  return bulk_sender{std::move(snd),shape,std::move(fun)};
}
}  // namespace Kokkos::StdExec
