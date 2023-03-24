#include <stdexec/execution.hpp>

namespace Kokkos::StdExec {

namespace impl_kokkos_scheduler {
template<class ExecutionSpace, int level=1>
class kokkos_scheduler {

  ExecutionSpace exec;

  template <class R>
  struct _op {
    [[no_unique_address]] R rec_;
    KOKKOS_FUNCTION
    friend void tag_invoke(stdexec::start_t, _op& op) noexcept 
    {
      stdexec::set_value((R &&) op.rec_);
    }
  };

  struct _attrs {
    kokkos_scheduler sch;
    template <class Tag>
    friend kokkos_scheduler tag_invoke(stdexec::get_completion_scheduler_t<Tag>,
                                       _attrs attr) noexcept {
      return attr.sch;
    }
  };

  struct _sender {
    kokkos_scheduler sch;

    using completion_signatures = stdexec::completion_signatures<
        stdexec::set_value_t(), stdexec::set_error_t(std::exception_ptr)>;

    template <class R>
    friend auto tag_invoke(stdexec::connect_t, _sender, R&& rec)
        // noexcept(std::is_nothrow_constructible_v<std::remove_cvref_t<R>, R>)
        -> _op<std::remove_cvref_t<R>> {
      return {(R &&) rec};
    }

    friend _attrs tag_invoke(stdexec::get_env_t, _sender snd) noexcept {
      return {snd.sch};
    }
  };

  friend _sender tag_invoke(stdexec::schedule_t,
                            const kokkos_scheduler& sch) noexcept {
    return {sch};
  }

 public:
  using execution_space = ExecutionSpace;

  kokkos_scheduler():exec(ExecutionSpace()) {}
  kokkos_scheduler(ExecutionSpace exec_):exec(exec_) {}

  ExecutionSpace get_execution_space() const { return exec; }

  auto bulk_policy(int N)
  requires(std::is_same_v<ExecutionSpace,Kokkos::Cuda>)
  {
    return Kokkos::RangePolicy<ExecutionSpace>(exec,0,N);
  }
  auto bulk_policy(int N)
  requires(!std::is_same_v<ExecutionSpace,Kokkos::Cuda>)
  {
    return Kokkos::TeamVectorRange(exec,0,N);
  }
  auto fence(const char* str)
  requires(std::is_same_v<ExecutionSpace,Kokkos::Cuda>)
  {
    return exec.fence(str);
  }
  auto fence(const char* str)
  requires(!std::is_same_v<ExecutionSpace,Kokkos::Cuda>)
  {
    return exec.team_barrier();
  }


  bool operator==(const kokkos_scheduler&) const noexcept = default;
};

template <class ReceiverId, std::integral Shape, class Fun, class Scheduler>
struct bulk_receiver {
  using is_receiver = void;

  [[no_unique_address]] Shape shape;
  [[no_unique_address]] Fun fun;
  [[no_unique_address]] ReceiverId successor;
  [[no_unique_address]] Scheduler sched;

  KOKKOS_FUNCTION
  friend void tag_invoke(stdexec::set_value_t, bulk_receiver&& self) noexcept 
  {
    KOKKOS_IF_ON_HOST(
    Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Cuda>(0,self.shape), self.fun);
    )
    KOKKOS_IF_ON_DEVICE(
                    printf("Exec\n");
      if constexpr (!std::is_same_v<typename Scheduler::execution_space,Kokkos::Cuda>) {
        Kokkos::parallel_for(self.sched.bulk_policy(self.shape), self.fun);
      }
    )
    stdexec::set_value(std::move(self.successor));
  }
  friend void tag_invoke(stdexec::set_error_t, bulk_receiver&& self,
                         std::exception_ptr&& except) noexcept {
    stdexec::set_error(std::move(self.successor), std::move(except));
  }
  explicit bulk_receiver(ReceiverId rcvr, Shape shape, Fun f, Scheduler sch)
      : shape(shape), fun((Fun &&) f), successor(rcvr), sched(sch) {}
};

template <class Sender, std::integral Shape, class Fun>
struct bulk_sender {
  template <stdexec::receiver Receiver, class Scheduler>
  using receiver_type = bulk_receiver<Receiver, Shape, Fun, Scheduler>;

  using is_sender = void;

  [[no_unique_address]] Sender sndr;
  [[no_unique_address]] Shape shape;
  [[no_unique_address]] Fun fun;

  using completion_signatures =
      stdexec::completion_signatures<stdexec::set_value_t(),
                                     stdexec::set_error_t(std::exception_ptr)>;

  template <stdexec::receiver Receiver>
  friend auto tag_invoke(stdexec::connect_t, bulk_sender&& self,
                         Receiver rcvr) noexcept {
    auto sched = stdexec::get_completion_scheduler<stdexec::set_value_t>(stdexec::get_env(self));
    return stdexec::connect(
        std::move(self.sndr),
        receiver_type<Receiver, decltype(sched)>{std::move(rcvr), self.shape,
                              std::move(self.fun),
                              std::move(sched)});
  }

  friend auto tag_invoke(stdexec::get_env_t,
                         const bulk_sender& self) noexcept {
    return stdexec::get_env(self.sndr);
  }
};

template <class ExecutionSpace, class Sender, std::integral Shape, class Fun>
auto tag_invoke(stdexec::bulk_t, kokkos_scheduler<ExecutionSpace>, Sender&& snd, Shape shape,
                Fun&& fun) {
  return bulk_sender{std::move(snd), shape, std::move(fun)};
}

template<class Scheduler>
struct sync_wait_receiver {
  using is_receiver = void;

  Scheduler sched;

  friend void tag_invoke(stdexec::set_value_t, sync_wait_receiver&& self) noexcept {
    self.sched.fence("sync_wait::fence");
  }
  friend void tag_invoke(stdexec::set_error_t, sync_wait_receiver&& self,
                         std::exception_ptr&& except) noexcept {
    Kokkos::abort("Error for sync_wait");
  }
  explicit sync_wait_receiver(Scheduler sched_):sched(sched_) {}
};

template <class ExecutionSpace, class Sender>
auto tag_invoke(stdexec::sync_wait_t, kokkos_scheduler<ExecutionSpace> sched, Sender&& snd) {
  auto op = stdexec::connect(std::move(snd), sync_wait_receiver(sched));
  stdexec::start(op);
}
} // namespace impl_kokkos_scheduler

using impl_kokkos_scheduler::kokkos_scheduler;
}  // namespace Kokkos::StdExec
