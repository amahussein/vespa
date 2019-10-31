package com.yahoo.container.core.config.testutil;

import com.yahoo.component.ComponentSpecification;
import com.yahoo.osgi.OsgiWrapper;
import org.osgi.framework.Bundle;

import java.util.Collection;
import java.util.List;

import static java.util.Collections.emptyList;

/**
 * @author gjoranv
 */
public class MockOsgiWrapper implements OsgiWrapper {

    @Override
    public List<Bundle> getInitialBundles() {
        return emptyList();
    }

    @Override
    public Bundle[] getBundles() {
        return new Bundle[0];
    }

    @Override
    public List<Bundle> getCurrentBundles() {
        return emptyList();
    }

    @Override
    public Bundle getBundle(ComponentSpecification bundleId) {
        return null;
    }

    @Override
    public List<Bundle> install(String absolutePath) {
        return emptyList();
    }

    @Override
    public void allowDuplicateBundles(Collection<Bundle> bundles) {  }
}