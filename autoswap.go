// Package autoswap provides a generic hot-reloader for file- or
// directory-based resources. It watches filesystem changes, detects content
// updates via a SHA-256 hash, and atomically swaps in new parsed values
// of an arbitrary type T. It supports debounced file events, versioning,
// and user-defined hooks.
package autoswap

import (
	"context"
	"crypto/sha256"
	"fmt"
	"io/fs"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"log/slog"

	"github.com/fsnotify/fsnotify"
)

// LoaderFunc is a user-supplied function that parses raw filesystem contents
// into a value of type T. The files map keys are relative paths within the
// watched filesystem and values are the corresponding file byte contents.
// LoaderFunc should return the parsed value or an error if parsing fails.
type LoaderFunc[T any] func(files map[string][]byte) (*T, error)

// Hooks groups user-defined callbacks for successful updates or errors.
// OnUpdate is invoked with the new value after each successful reload.
// OnError is invoked with the error encountered during TryUpdate or Start.
type Hooks[T any] struct {
	OnUpdate func(value *T)
	OnError  func(err error)
}

// Watcher monitors a file or directory and reloads values of type T when
// changes occur. It uses an atomic pointer for safe concurrent access,
// a SHA-256 hash to detect content changes, and an optional debounce period
// to coalesce rapid filesystem events.
type Watcher[T any] struct {
	fs       fs.FS         // filesystem to read from
	origPath string        // original path (file or directory)
	path     string        // internal root path (file or "." for dir)
	label    string        // identifier for logging
	loader   LoaderFunc[T] // parsing function
	hooks    Hooks[T]      // user callbacks
	debounce time.Duration // debounce duration for events
	isDir    bool          // true if watching a directory

	val     atomic.Pointer[T]        // holds the current value
	hash    atomic.Pointer[[32]byte] // holds the last computed hash
	version atomic.Uint64            // update counter
}

// New constructs a Watcher for a single file.
// fsys is the filesystem to read from (e.g., os.DirFS("/")); path is the file
// path to watch; label is used in logs; loader parses file contents into *T;
// hooks define optional OnUpdate and OnError callbacks.
func New[T any](fsys fs.FS, path, label string, loader LoaderFunc[T], hooks Hooks[T]) *Watcher[T] {
	return &Watcher[T]{
		fs:       fsys,
		origPath: path,
		path:     path,
		label:    label,
		loader:   loader,
		hooks:    hooks,
		debounce: 200 * time.Millisecond,
		isDir:    false,
	}
}

// NewDirWatcher constructs a Watcher for all files under a directory.
// fsys is the base filesystem; dir is the directory path to watch; label
// is used in logs; loader parses the collected files map into *T;
// hooks define callbacks. It scopes the FS via fs.Sub if supported.
func NewDirWatcher[T any](fsys fs.FS, dir, label string, loader LoaderFunc[T], hooks Hooks[T]) *Watcher[T] {
	sub, err := fs.Sub(fsys, dir)
	if err != nil {
		sub = fsys // fallback to original FS
	}
	return &Watcher[T]{
		fs:       sub,
		origPath: dir,
		path:     ".",
		label:    label,
		loader:   loader,
		hooks:    hooks,
		debounce: 200 * time.Millisecond,
		isDir:    true,
	}
}

// Load returns the most recently loaded value of type T, or nil if not loaded.
// Safe for concurrent use.
func (w *Watcher[T]) Load() *T {
	return w.val.Load()
}

// Version returns a monotonically increasing counter that increments on each
// successful reload. Useful for cache invalidation or change tracking.
func (w *Watcher[T]) Version() uint64 {
	return w.version.Load()
}

// Sync forces a reload: it reads the watched file or directory,
// computes a combined SHA-256 hash, and if the hash differs from the
// previous version, parses new content via the loader, updates the value,
// increments the version, and invokes OnUpdate. If an error occurs,
// OnError is invoked. Returns any I/O or parsing error.
func (w *Watcher[T]) Sync() error {
	files := make(map[string][]byte)

	// Collect file contents
	if w.isDir {
		if err := fs.WalkDir(w.fs, w.path, func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return fmt.Errorf("error walking directory %s: %w", w.path, err)
			}
			if d.IsDir() {
				return nil
			}
			if data, err := fs.ReadFile(w.fs, p); err == nil {
				files[p] = data
			}
			return nil
		}); err != nil {
			if w.hooks.OnError != nil {
				w.hooks.OnError(err)
			}
			return err
		}
	} else {
		data, err := fs.ReadFile(w.fs, w.path)
		if err != nil {
			if w.hooks.OnError != nil {
				w.hooks.OnError(err)
			}
			return err
		}
		files[w.path] = data
	}

	// Compute deterministic hash over filenames and contents
	h := sha256.New()
	keys := make([]string, 0, len(files))
	for k := range files {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		h.Write([]byte(k))
		h.Write(files[k])
	}
	var sum [32]byte
	copy(sum[:], h.Sum(nil))

	// No change detected
	if prev := w.hash.Load(); prev != nil && *prev == sum {
		return nil
	}

	// Parse and store new value
	val, err := w.loader(files)
	if err != nil {
		if w.hooks.OnError != nil {
			w.hooks.OnError(err)
		}
		return err
	}

	w.val.Store(val)
	w.hash.Store(&sum)
	newVer := w.version.Add(1)

	if w.hooks.OnUpdate != nil {
		w.hooks.OnUpdate(val)
	}

	slog.Info("autoswap update", slog.String("label", w.label), slog.String("path", w.origPath), slog.Uint64("version", newVer))
	return nil
}

// Watch begins monitoring the file or directory and triggers TryUpdate()
// after a debounce delay when relevant events occur. The watcher runs in a
// background goroutine and stops when ctx is cancelled. Returns an error
// if the internal file watcher cannot be created or initialized.
func (w *Watcher[T]) Watch(ctx context.Context) error {
	fw, err := fsnotify.NewWatcher()
	if err != nil {
		return err
	}
	dirToWatch := w.origPath
	if !w.isDir {
		dirToWatch = filepath.Dir(w.origPath)
	}
	if err := fw.Add(dirToWatch); err != nil {
		fw.Close()
		return err
	}

	go func() {
		defer fw.Close()
		var timer *time.Timer
		var timerMu sync.Mutex // Protects access to the timer
		for {
			select {
			case <-ctx.Done():
				return
			case ev := <-fw.Events:
				match := false
				if w.isDir {
					if (ev.Op&fsnotify.Write != 0 || ev.Op&fsnotify.Create != 0) &&
						strings.HasPrefix(filepath.Clean(ev.Name), filepath.Clean(w.origPath)) {
						match = true
					}
				} else {
					if (ev.Op&fsnotify.Write != 0 || ev.Op&fsnotify.Create != 0) &&
						filepath.Clean(ev.Name) == filepath.Clean(w.origPath) {
						match = true
					}
				}
				if match {
					timerMu.Lock()
					if timer != nil {
						timer.Reset(w.debounce)
					} else {
						timer = time.AfterFunc(w.debounce, func() {
							if err := w.Sync(); err != nil {
								slog.Info("autoswap event", slog.String("label", w.label), slog.String("event", ev.Op.String()), slog.String("path", ev.Name))
								if w.hooks.OnError != nil {
									w.hooks.OnError(err)
								}
							}
						})
					}
					timerMu.Unlock()
				}
			case err := <-fw.Errors:
				slog.Warn("autoswap watcher error", slog.String("label", w.label), slog.String("error", err.Error()))
			}
		}
	}()

	return nil
}

// Start performs an initial TryUpdate and then begins watching for changes.
// Returns any error from the initial load or watcher setup.
func (w *Watcher[T]) Start(ctx context.Context) error {
	if err := w.Sync(); err != nil {
		return err
	}
	return w.Watch(ctx)
}
